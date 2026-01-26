for seed in 0 1 2 3; do
  for task in task1; do
    for agent_name in fan; do
      for alpha1 in 100; do
        for alpha2 in 0.1; do
          echo ""
          echo "--- Running task: $task with (alpha1,alpha2): ($alpha1,$alpha2) ---"
          python main.py \
            --seed=${seed} \
            --train_steps=500000 \
            --env_name=visual-cube-double-play-singletask-${task}-v0 \
            --agent=agents/${agent_name}.py \
            --agent.alpha1=${alpha1} \
            --agent.alpha2=${alpha2} \
            --agent.z_agg=min \
            --agent.discount=0.995 \
            --p_aug=0.5 --frame_stack=3 --agent.encoder=impala_small
          echo ""
        done
      done
    done
  done
done