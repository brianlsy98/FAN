'''
NFQL (Noise-conditioned Flow Q-learning):
 - Behavior Regularization : Flow Distillation (in FQL)
 - Value Function Update   : Noise-conditioned Q-learning
'''
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class NFQLAgent(flax.struct.PyTreeNode):
    """Noise-conditioned Flow Q-learning (NFQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile, power=2):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**power)
    
    def critic_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        
        """Compute the TD critic loss."""
        ### Q loss.
        # Actor Next Actions.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_next_actions = self.network.select('actor_onestep_flow')(batch['next_observations'], noises)
        actor_next_actions = jnp.clip(actor_next_actions, -1, 1)

        # Next Z.
        next_zs = self.network.select('critic_z')(batch['next_observations'], actions=actor_next_actions)
        if self.config['z_agg'] == 'min':
            next_z = next_zs.min(axis=0)
        else:
            next_z = next_zs.mean(axis=0)

        # Target Q.
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_z
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], noises=noises, params=grad_params)
        q_loss = jnp.square(q - target_q).mean()

        ### Z loss.
        z = self.network.select('critic_z')(batch['observations'], actions=batch['actions'], params=grad_params)        # (2, B)
        rng, n_rng = jax.random.split(rng)
        n = jax.random.normal(n_rng, (batch_size, action_dim))
        target_zs = self.network.select('target_critic')(batch['observations'], actions=batch['actions'], noises=n)     # (2, B)
        target_z = target_zs.mean(axis=0)
        z_loss = self.expectile_loss(target_z - z, target_z - z, expectile=self.config['expectile']).mean()

        critic_loss = q_loss + z_loss
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_loss': q_loss,
            'z_loss': z_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'z_mean': z.mean(),
            'z_max': z.max(),
            'z_min': z.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the one-step flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        ### BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        vel = x_1 - x_0
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        br_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)

        ### Q loss.
        rng, n_rng = jax.random.split(rng)
        n = jax.random.normal(n_rng, (batch_size, action_dim))
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions, noises=n)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_value_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        ### Z loss.
        zs = self.network.select('critic_z')(batch['observations'], actions=actor_actions)
        z = jnp.mean(zs, axis=0)
        z_loss = -z.mean()
        if self.config['normalize_value_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(z).mean())
            z_loss = lam * z_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * br_loss + q_loss + z_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'br_loss': br_loss,
            'q_loss': q_loss,
            'z_loss': z_loss,
            'q': q.mean(),
            'z': z.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        return actions


    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions


    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_z'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_z_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic_z'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic_z=(critic_z_def, (ex_observations, ex_actions)),
            critic=(critic_def, (ex_observations, ex_actions, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='nfql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),    # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,           # Learning rate.
            batch_size=256,    # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,   # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.995,    # Discount factor.
            tau=0.005,         # Target network update rate.
            z_agg='mean',      # Aggregation method for target Z values.
            alpha=10.0,        # BC coefficient (need to be tuned for each environment).
            flow_steps=10,     # Number of flow steps.
            expectile=0.9,     # Expectile for loss computation.
            normalize_value_loss=False,  # Whether to normalize the value loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
