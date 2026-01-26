for seed in 0 1 2 3 4 5 6 7; do
  for env_name in antmaze-medium-play-v2  antmaze-medium-diverse-v2 antmaze-large-play-v2 antmaze-large-diverse-v2; do
    for agent_name in fan; do
      for alpha1 in 3; do
        for alpha2 in 0.01 0.03; do
          echo ""
          echo "--- Running env: $env_name with (alpha1,alpha2): ($alpha1,$alpha2) ---"
          python main.py \
            --seed=${seed} \
            --train_steps=500000 \
            --env_name=${env_name} \
            --agent=agents/${agent_name}.py \
            --agent.alpha1=${alpha1} \
            --agent.alpha2=${alpha2} \
            --agent.z_agg=min \
            --agent.discount=0.99
          echo ""
        done
      done
    done
  done
done