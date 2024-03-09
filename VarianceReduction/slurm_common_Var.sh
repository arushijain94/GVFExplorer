#!/bin/bash

alphas=(0.15 0.25 0.5)
exploration_steps_list=(4000)
#env_name_list=('online_rooms_same_reward_grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')
num_iter=10000
temp_name='28Feb'
#jobname='Var'
#jobname='VarGVF'
jobname='Var_Greedy'

for env_name in "${env_name_list[@]}"
do
  for exploration_steps in "${exploration_steps_list[@]}"
  do
    for alpha in "${alphas[@]}"
    do
      for seed in {1..1}
      do
        echo "sbatch common_var_script.sh $alpha $seed $num_iter $temp_name $env_name $exploration_steps"
        sbatch -J $jobname ./common_var_script.sh $alpha $seed $num_iter $temp_name $env_name $exploration_steps
      done
    done
  done
done
