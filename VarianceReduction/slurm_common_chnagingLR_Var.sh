#!/bin/bash
max_alpha_Q_list=(1.)
min_alpha_Q_list=(0.5)
max_alpha_var_list=(1.0)
min_alpha_var_list=(0.8)
lr_decay_steps_Q_list=(500000)
lr_decay_steps_var_list=(500000)
exploration_steps_list=(500)
#lr_decay_steps_var=10000
#env_name_list=('online_rooms_same_reward_grid')
#env_name_list=('online_rooms_different_reward_grid')
#env_name_list=('semigreedy_online_rooms_different_reward_grid')
env_name_list=('semigreedy_online_rooms_different_reward_grid')
num_iter=30000
temp_name='7mar_final'

jobname='Var'
#jobname='VarGVF'
#jobname='Var_Greedy'x
for lr_decay_steps_var in "${lr_decay_steps_var_list[@]}"
do
  for max_alpha_var in "${max_alpha_var_list[@]}"
  do
    for max_alpha_Q in "${max_alpha_Q_list[@]}"
    do
      for lr_decay_steps_Q in "${lr_decay_steps_Q_list[@]}"
      do
        for min_alpha_var in "${min_alpha_var_list[@]}"
        do
          for env_name in "${env_name_list[@]}"
          do
            for exploration_steps in "${exploration_steps_list[@]}"
            do
              for min_alpha_Q in "${min_alpha_Q_list[@]}"
              do
                for seed in {1..25}
                do
                  echo "sbatch run_common_var_changing_lr.sh $min_alpha_Q $seed $num_iter $temp_name $env_name $exploration_steps $lr_decay_steps_Q $min_alpha_var $lr_decay_steps_var $max_alpha_var $max_alpha_Q"
                  sbatch -J $jobname ./run_common_var_changing_lr.sh $min_alpha_Q $seed $num_iter $temp_name $env_name $exploration_steps $lr_decay_steps_Q $min_alpha_var $lr_decay_steps_var $max_alpha_var $max_alpha_Q
                done
              done
            done
          done
        done
      done
    done
  done
done