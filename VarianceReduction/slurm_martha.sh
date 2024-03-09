#!/bin/bash

start_types=(2)
alphas=(0.8)
type_behavior_policys=(0)
num_eps=(1)
jobname='SR'
exploration_steps_list=(200 4000)
behv_reward_combined_list=(0)

for behv_reward_combined in "${behv_reward_combined_list[@]}"
do
  for exploration_steps in "${exploration_steps_list[@]}"
    do
    for num_ep in "${num_eps[@]}"
    do
      for type_behavior_policy in "${type_behavior_policys[@]}"
      do
        for start_type in "${start_types[@]}"
        do
          for alpha in "${alphas[@]}"
          do
            for seed in {1..1}
            do
              echo "sbatch run_martha.sh $alpha $seed $start_type $type_behavior_policy $num_ep $exploration_steps $behv_reward_combined"
              sbatch -J $jobname ./run_martha.sh $alpha $seed $start_type $type_behavior_policy $num_ep $exploration_steps $behv_reward_combined
            done
          done
        done
      done
    done
  done
done