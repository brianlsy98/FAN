for seed in 0 1 2 3 4 5 6 7; do
  for task in task4; do
    for agent_name in nbrac; do
      for alpha1 in 100; do
        for alpha2 in 3; do
          echo ""
          echo "--- Running task: $task with (alpha1,alpha2): ($alpha1,$alpha2) ---"
          python main.py \
            --seed=${seed} \
            --env_name=puzzle-3x3-play-singletask-${task}-v0 \
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
