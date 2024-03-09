#!/bin/bash

num_epsiodes=(1)
min_alpha_list=(0.9)
exploration_steps_list=(500)
#env_name_list=('online_rooms_same_reward_grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')
num_iter=90000
temp_name='7march_final'
jobname='Jos'


for env_name in "${env_name_list[@]}"
do
  for num_ep in "${num_epsiodes[@]}"
  do
    for min_alpha in "${min_alpha_list[@]}"
    do
      for exploration_steps in "${exploration_steps_list[@]}"
      do
        for seed in {1..25}
        do
          echo "sbatch common_josiah_changing_lr.sh $min_alpha $seed $num_iter $temp_name $env_name $exploration_steps $num_ep"
          sbatch -J $jobname ./common_josiah_changing_lr.sh $min_alpha $seed $num_iter $temp_name $env_name $exploration_steps $num_ep
        done
      done
    done
  done
done



