#!/bin/bash

lr_decay_steps_list=(500000)
min_alpha_list=(0.8)
type_behavior_policys=(1)
exploration_steps_list=(500)
behv_reward_combined_list=(1)
#env_name_list=('online_rooms_same_reward_grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')
num_iter=60000
temp_name='7march_final_2'
#jobname='SR'
#jobname='SRGVF'
jobname='SR'


for lr_decay_steps in "${lr_decay_steps_list[@]}"
do
  for exploration_steps in "${exploration_steps_list[@]}"
  do
    for env_name in "${env_name_list[@]}"
    do
      for type_behavior_policy in "${type_behavior_policys[@]}"
      do
        for behv_reward_combined in "${behv_reward_combined_list[@]}"
        do
          for min_alpha in "${min_alpha_list[@]}"
          do
            for seed in {1..25}
            do
              echo "sbatch run_common_martha_changing_lr.sh $min_alpha $seed $env_name $type_behavior_policy $temp_name $exploration_steps $behv_reward_combined $num_iter $lr_decay_steps"
              sbatch -J $jobname ./run_common_martha_changing_lr.sh $min_alpha $seed $env_name $type_behavior_policy $temp_name $exploration_steps $behv_reward_combined $num_iter $lr_decay_steps
            done
          done
        done
      done
    done
  done
done
