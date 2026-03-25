for seed in 0; do
  for task in task2 task3 task4 task5; do
    for agent_name in fan; do
      for alpha1 in 100; do
        for alpha2 in 3; do
          for expectile in 0.5 0.7 0.9 0.99 1.0; do
            for er_coeff in 1.0; do
              for zmax_coeff in 1.0; do
                echo ""
                echo "--- Running task: $task with (alpha1,alpha2,expectile,er_coeff,zmax_coeff): ($alpha1,$alpha2,$expectile,$er_coeff,$zmax_coeff) ---"
                python main.py \
                  --seed=${seed} \
                  --env_name=puzzle-4x4-play-singletask-${task}-v0 \
                  --agent=agents/${agent_name}.py \
                  --agent.alpha1=${alpha1} \
                  --agent.alpha2=${alpha2} \
                  --agent.expectile=${expectile} \
                  --agent.er_coeff=${er_coeff} \
                  --agent.zmax_coeff=${zmax_coeff} \
                  --agent.discount=0.995
                echo ""
              done
            done
          done
        done  
      done
    done
  done  
done