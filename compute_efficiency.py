# Benchmark script for measuring FLOPs and runtime.
"""
Measure FLOPs and wall-clock time for JAX agents.

Reports (per agent):
  - Training FLOPs   (GFLOPs)
  - Training runtime
  - Inference FLOPs  (MFLOPs)
  - Inference runtime
"""


from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import jax
import jax.numpy as jnp


# ============================================================
# Agent registry
# ============================================================

AGENTS = [
    ("FAN", "fan.py", "FANAgent"),
    ("NBRAC", "nbrac.py", "NBRACAgent"),
    ("NFQL", "nfql.py", "NFQLAgent"),
    ("FAQL", "faql.py", "FAQLAgent"),
]


# ============================================================
# FLOPs helpers
# ============================================================

def _extract_flops(cost):
    if cost is None:
        return None
    if isinstance(cost, dict):
        f = cost.get("flops", None)
        return None if f is None or f < 0 else float(f)
    if isinstance(cost, (list, tuple)):
        total = 0.0
        found = False
        for c in cost:
            if isinstance(c, dict) and "flops" in c and c["flops"] >= 0:
                total += float(c["flops"])
                found = True
        return total if found else None
    return None


def estimate_flops(jitted_fn, *abstract_args):
    lowered = jitted_fn.lower(*abstract_args)
    compiled = lowered.compile()
    cost = compiled.cost_analysis()
    return _extract_flops(cost)


# ============================================================
# Runtime helpers
# ============================================================

def measure_time(jitted_fn, real_args, num_warmup=5, num_runs=50):
    # warmup
    for _ in range(num_warmup):
        out = jitted_fn(*real_args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

    # timed
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out = jitted_fn(*real_args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t1 = time.perf_counter()

    return (t1 - t0) / num_runs


# ============================================================
# Utilities
# ============================================================

def import_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_example_batch(bs, obs_dim, act_dim):
    return {
        "observations": jnp.zeros((bs, obs_dim), jnp.float32),
        "actions": jnp.zeros((bs, act_dim), jnp.float32),
        "min_reward": -1.0,
        "max_reward": 1.0,
    }


def make_train_batch_struct(bs, obs_dim, act_dim):
    f32 = jnp.float32
    return {
        "observations": jax.ShapeDtypeStruct((bs, obs_dim), f32),
        "actions": jax.ShapeDtypeStruct((bs, act_dim), f32),
        "next_observations": jax.ShapeDtypeStruct((bs, obs_dim), f32),
        "next_actions": jax.ShapeDtypeStruct((bs, act_dim), f32),
        "rewards": jax.ShapeDtypeStruct((bs,), f32),
        "masks": jax.ShapeDtypeStruct((bs,), f32),
    }


def make_train_batch_real(bs, obs_dim, act_dim):
    return {
        "observations": jnp.zeros((bs, obs_dim), jnp.float32),
        "actions": jnp.zeros((bs, act_dim), jnp.float32),
        "next_observations": jnp.zeros((bs, obs_dim), jnp.float32),
        "next_actions": jnp.zeros((bs, act_dim), jnp.float32),
        "rewards": jnp.zeros((bs,), jnp.float32),
        "masks": jnp.ones((bs,), jnp.float32),
    }


def force_encoder_none(cfg):
    if hasattr(cfg, "encoder"):
        cfg.encoder = None
    return cfg



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents_dir", type=str, default="./agents")
    parser.add_argument("--obs_dim", type=int, default=37)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agents_dir = Path(args.agents_dir)

    example_batch = make_example_batch(1, args.obs_dim, args.action_dim)
    train_batch_struct = make_train_batch_struct(args.batch, args.obs_dim, args.action_dim)
    train_batch_real = make_train_batch_real(args.batch, args.obs_dim, args.action_dim)

    obs_real = jnp.zeros((args.obs_dim,), jnp.float32)
    key_real = jax.random.PRNGKey(0)

    obs_struct = jax.ShapeDtypeStruct((args.obs_dim,), jnp.float32)
    key_struct = jax.ShapeDtypeStruct((2,), jnp.uint32)

    results = []

    for name, fname, cls in AGENTS:
        mod = import_module_from_file(agents_dir / fname)
        Agent = getattr(mod, cls)
        cfg = force_encoder_none(mod.get_config())
        agent = Agent.create(args.seed, example_batch, cfg)

        # ---------- training ----------
        if name == "ReBRAC":
            def _update(b): return agent.update(b, full_update=True)
        else:
            def _update(b): return agent.update(b)

        update_fn = jax.jit(_update)
        upd_flops = estimate_flops(update_fn, train_batch_struct)
        upd_time = measure_time(update_fn, (train_batch_real,), num_runs=args.runs)

        # ---------- inference ----------
        def _sample(o, k): return agent.sample_actions(o, k)

        sample_fn = jax.jit(_sample)
        samp_flops = estimate_flops(sample_fn, obs_struct, key_struct)
        samp_time = measure_time(sample_fn, (obs_real, key_real), num_runs=args.runs)

        results.append(
            (name, upd_time, upd_flops, samp_time, samp_flops)
        )

    # Print once at the end
    print(
        "\nAgent | "
        "Upd ms | Upd GFLOPs | "
        "Samp ms | Samp MFLOPs"
    )
    print("-" * 80)

    for name, ut, uf, st, sf in results:
        print(
            f"{name:<8} | "
            f"{ut*1e3:6.2f} | {uf/1e9:10.3f} | "
            f"{st*1e3:7.2f} | {sf/1e6:10.3f}"
        )

if __name__ == "__main__":
    main()
