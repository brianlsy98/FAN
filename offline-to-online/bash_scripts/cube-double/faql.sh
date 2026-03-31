for seed in 0 1 2 3 4 5 6 7; do
  for agent_name in faql; do
    for alpha1 in 100; do
      for alpha2 in 0; do
        for online_alpha1 in 30; do
          for online_alpha2 in 0; do
            echo ""
            echo "--- cube-double-play-singletask-v0 with (a1,a2,o1,o2): ($alpha1,$alpha2,$online_alpha1,$online_alpha2) ---"
            python offline-to-online/offline_to_online.py \
              --seed=${seed} \
              --env_name="cube-double-play-singletask-v0" \
              --agent=agents/${agent_name}.py \
              --agent.alpha1=${alpha1} \
              --agent.alpha2=${alpha2} \
              --agent.discount=0.995 \
              --agent.online_alpha1=${online_alpha1} \
              --agent.online_alpha2=${online_alpha2} \
              --online_steps=1000000
            echo ""
          done
        done
      done
    done
  done
done