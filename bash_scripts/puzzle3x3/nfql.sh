for seed in 0 1 2 3 4 5 6 7; do
  for task in task4; do
    for agent_name in nfql; do
      for alpha in 1000; do
        echo ""
        echo "--- Running task: $task with (alpha): ($alpha) ---"
        python main.py \
          --seed=${seed} \
          --env_name=puzzle-3x3-play-singletask-${task}-v0 \
          --agent=agents/${agent_name}.py \
          --agent.alpha=${alpha} \
          --agent.expectile=0.9 \
          --agent.discount=0.995
        echo ""
      done
    done
  done
done