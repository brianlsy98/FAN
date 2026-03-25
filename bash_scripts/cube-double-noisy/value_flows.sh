for seed in 0 1 2; do
  for task in task1; do
    for agent_name in value_flows; do
      for bcfm_lambda in 1.0; do
        for confidence_weight_temp in 3.0; do
          echo ""
          echo "--- Running task: $task with (agent_name, bcfm_lambda, confidence_weight_temp): ($agent_name, $bcfm_lambda, $confidence_weight_temp) ---"
          python main.py \
            --seed=${seed} \
            --env_name=cube-double-noisy-singletask-${task}-v0 \
            --agent=agents/${agent_name}.py \
            --agent.bcfm_lambda=${bcfm_lambda} \
            --agent.confidence_weight_temp=${confidence_weight_temp} \
            --agent.discount=0.995
          echo ""
        done
      done
    done
  done
done