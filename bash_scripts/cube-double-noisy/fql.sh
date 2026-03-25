for seed in 0 1 2; do
  for task in task1; do
    for agent_name in fql; do
      for alpha in 100; do
        echo ""
        echo "--- Running task: $task with (alpha): ($alpha) ---"
        python main.py \
          --seed=${seed} \
          --env_name=cube-double-noisy-singletask-${task}-v0 \
          --agent=agents/${agent_name}.py \
          --agent.alpha=${alpha} \
          --agent.discount=0.995
        echo ""
      done
    done
  done
done