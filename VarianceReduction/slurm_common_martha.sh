#!/bin/bash

alphas=(0.5)
type_behavior_policys=(0)
exploration_steps_list=(4000)
behv_reward_combined_list=(1)
#env_name_list=('online_rooms_same_reward_grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')
num_iter=18000
temp_name='28Feb'
#jobname='SR'
#jobname='SRGVF'
jobname='SR_Greedy'

for exploration_steps in "${exploration_steps_list[@]}"
do
  for env_name in "${env_name_list[@]}"
  do
    for type_behavior_policy in "${type_behavior_policys[@]}"
    do
      for behv_reward_combined in "${behv_reward_combined_list[@]}"
      do
        for alpha in "${alphas[@]}"
        do
          for seed in {1..1}
          do
            echo "sbatch common_martha_script.sh $alpha $seed $env_name $type_behavior_policy $temp_name $exploration_steps $behv_reward_combined $num_iter"
            sbatch -J $jobname ./common_martha_script.sh $alpha $seed $env_name $type_behavior_policy $temp_name $exploration_steps $behv_reward_combined $num_iter
          done
        done
      done
    done
  done
done
