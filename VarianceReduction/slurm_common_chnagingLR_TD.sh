#!/bin/bash

type_tars=(0)
min_alphas=(0.95)
lr_steps_list=(500000)
exploration_steps_list=(500)

#env_name_list=('online_rooms_same_reward_ grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')

num_iter=80000
temp_name='7march_final4'

jobname='TD'
#jobname='TDGVF'
#jobname='TD_Greedy'

for lr_step in "${lr_steps_list[@]}"
do
  for env_name in "${env_name_list[@]}"
  do
    for type_tar in "${type_tars[@]}"
    do
      for min_alpha in "${min_alphas[@]}"
      do
          for exploration_steps in "${exploration_steps_list[@]}"
          do
            for seed in {1..25}
            do
              echo "sbatch run_common_td_chnaging_lr.sh $min_alpha $exploration_steps $seed $type_tar $env_name $temp_name $num_iter $lr_step"
              sbatch -J $jobname ./run_common_td_chnaging_lr.sh $min_alpha $exploration_steps $seed $type_tar $env_name $temp_name $num_iter $lr_step
            done
          done
      done
    done
  done
done


