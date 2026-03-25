for seed in 0 1; do
  for task in task1; do
    for agent_name in fan; do
      for alpha1 in 100; do
        for alpha2 in 3; do
          for expectile in 0.5 0.7 0.9 0.99 1.0; do
            echo ""
            echo "--- Running task: $task with (alpha1,alpha2,expectile): ($alpha1,$alpha2,$expectile) ---"
            python main.py \
              --seed=${seed} \
              --env_name=scene-play-singletask-${task}-v0 \
              --agent=agents/${agent_name}.py \
              --agent.alpha1=${alpha1} \
              --agent.alpha2=${alpha2} \
              --agent.expectile=${expectile} \
              --agent.discount=0.995
            echo ""
          done
        done
      done
    done
  done
done
