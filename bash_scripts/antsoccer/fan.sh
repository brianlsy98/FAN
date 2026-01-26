for seed in 0 1 2 3 4 5 6 7; do
  for task in task1 task2 task3 task4 task5; do
    for agent_name in fan; do
      for alpha1 in 10; do
        for alpha2 in 0.1; do
          echo ""
          echo "--- Running task: $task with (alpha1,alpha2): ($alpha1,$alpha2) ---"
          python main.py \
            --seed=${seed} \
            --env_name=antsoccer-arena-navigate-singletask-${task}-v0 \
            --agent=agents/${agent_name}.py \
            --agent.alpha1=${alpha1} \
            --agent.alpha2=${alpha2} \
            --agent.expectile=0.9 \
            --agent.discount=0.995
          echo ""
        done
      done
    done
  done
done