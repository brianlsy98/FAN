for seed in 0 1; do
  for env_name in antmaze-medium-play-v2  antmaze-medium-diverse-v2; do
    for agent_name in fan; do
      for alpha1 in 3; do
        for alpha2 in 0.01; do
          for expectile in 0.5 0.7 0.9 0.99 1.0; do
            echo ""
            echo "--- Running env: $env_name with (alpha1,alpha2,expectile): ($alpha1,$alpha2,$expectile) ---"
            python main.py \
              --seed=${seed} \
              --train_steps=500000 \
              --env_name=${env_name} \
              --agent=agents/${agent_name}.py \
              --agent.alpha1=${alpha1} \
              --agent.alpha2=${alpha2} \
              --agent.z_agg=min \
              --agent.expectile=${expectile} \
              --agent.discount=0.99
            echo ""
          done
        done
      done
    done
  done
done

for seed in 0 1; do
  for env_name in antmaze-large-play-v2 antmaze-large-diverse-v2; do
    for agent_name in fan; do
      for alpha1 in 3; do
        for alpha2 in 0.03; do
          for expectile in 0.5 0.7 0.9 0.99 1.0; do
          echo ""
          echo "--- Running env: $env_name with (alpha1,alpha2,expectile): ($alpha1,$alpha2,$expectile) ---"
            python main.py \
              --seed=${seed} \
              --train_steps=500000 \
              --env_name=${env_name} \
              --agent=agents/${agent_name}.py \
              --agent.alpha1=${alpha1} \
              --agent.alpha2=${alpha2} \
              --agent.z_agg=min \
              --agent.expectile=${expectile} \
              --agent.discount=0.99
            echo ""
        done
      done
    done
  done
done