#!/bin/bash

start_type=2
min_alpha_q_list=(0.5)
min_alpha_var_list=(0.95)
num_iter=30000

jobname='Var_scaled'
temp_name="4march"


for min_alpha_var in "${min_alpha_var_list[@]}"
do
  for min_alpha_Q in "${min_alpha_q_list[@]}"
  do
    for seed in {1..25}
    do
      echo "sbatch run_scaled_bpi.sh $min_alpha_Q $seed $start_type $temp_name $min_alpha_var $num_iter"
      sbatch -J $jobname ./run_scaled_bpi.sh $min_alpha_Q $seed $start_type $temp_name $min_alpha_var $num_iter
    done
  done
done
