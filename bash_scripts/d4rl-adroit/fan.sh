for seed in 0 1 2 3 4 5 6 7; do
  for env_name in pen-human-v0 pen-cloned-v0 pen-expert-v0; do
    for agent_name in fan; do
      for alpha1 in 1000; do
        for alpha2 in 0 1; do
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

for seed in 0 1 2 3 4 5 6 7; do
  for env_name in door-human-v0 door-cloned-v0 door-expert-v0; do
    for agent_name in fan; do
      for alpha1 in 3000 10000; do
        for alpha2 in 10; do
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

for seed in 0 1 2 3 4 5 6 7; do
  for env_name in hammer-human-v0 hammer-cloned-v0 hammer-expert-v0; do
    for agent_name in fan; do
      for alpha1 in 10000; do
        for alpha2 in 0.3 1; do
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

for seed in 0 1 2 3 4 5 6 7; do
  for env_name in relocate-human-v0 relocate-cloned-v0 relocate-expert-v0; do
    for agent_name in fan; do
      for alpha1 in 10000 30000; do
        for alpha2 in 10; do
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