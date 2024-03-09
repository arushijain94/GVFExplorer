#!/bin/bash

type_target_list=(0 1 2)
min_alpha_list=(0.8)
temp_name="4march"
num_iter=20000

jobname='tdscale'

for min_alpha in "${min_alpha_list[@]}"
do
  for type_target in "${type_target_list[@]}"
  do
    for seed in {1..20}
    do
      echo "sbatch run_td_scaled.sh $min_alpha $seed $type_target $temp_name $num_iter"
      sbatch -J $jobname ./run_td_scaled.sh $min_alpha $seed $type_target $temp_name $num_iter
    done
  done
done
