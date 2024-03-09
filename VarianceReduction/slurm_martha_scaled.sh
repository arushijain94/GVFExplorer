#!/bin/bash

min_alpha_list=(0.25)
temp_name="4march_final"
num_iter=50000
jobname='SRscale'


for min_alpha in "${min_alpha_list[@]}"
do
  for seed in {1..20}
  do
    echo "sbatch run_martha_scaled.sh $min_alpha $seed $temp_name $num_iter"
    sbatch -J $jobname ./run_martha_scaled.sh $min_alpha $seed $temp_name $num_iter
  done
done

