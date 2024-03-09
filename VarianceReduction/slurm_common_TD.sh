#!/bin/bash

type_tars=(0 1 2)
alphas=(0.5)
exploration_steps_list=(4000)
#exploration_steps_list=(4000)
#env_name_list=('online_rooms_same_reward_grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')
num_iter=30000
temp_name='6march'
#jobname='TDGVF'
jobname='TD_Greedy'

for env_name in "${env_name_list[@]}"
do
  for type_tar in "${type_tars[@]}"
  do
    for alpha in "${alphas[@]}"
    do
        for exploration_steps in "${exploration_steps_list[@]}"
        do
          for seed in {1..20}
          do
            echo "sbatch common_td_script.sh $alpha $exploration_steps $seed $type_tar $env_name $temp_name $num_iter"
            sbatch -J $jobname ./common_td_script.sh $alpha $exploration_steps $seed $type_tar $env_name $temp_name $num_iter
          done
        done
    done
  done
done



