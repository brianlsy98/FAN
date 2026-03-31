for seed in 0 1 2 3 4 5 6 7; do
  for task in task4; do
    for agent_name in faql; do
      for alpha1 in 1000; do
        for alpha2 in 0; do
          echo ""
          echo "--- Running task: $task with (alpha1,alpha2): ($alpha1,$alpha2) ---"
          python main.py \
            --seed=${seed} \
            --env_name=puzzle-4x4-play-singletask-${task}-v0 \
            --agent=agents/${agent_name}.py \
            --agent.alpha1=${alpha1} \
            --agent.alpha2=${alpha2} \
            --agent.discount=0.995
          echo ""
        done
      done
    done
  done
done