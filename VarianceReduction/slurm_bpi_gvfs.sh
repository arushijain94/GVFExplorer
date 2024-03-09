#!/bin/bash

start_types=(2)
alphas=(0.8 0.25 0.1)
type_behavior_policys=(0)
num_eps=(1)
jobname='bpigvf'
exploration_steps_list=(4000)
exp_next_vals=(1)

for exp_next_val in "${exp_next_vals[@]}"
do
  for exploration_steps in "${exploration_steps_list[@]}"
    do
    for num_ep in "${num_eps[@]}"
    do
      for type_behavior_policy in "${type_behavior_policys[@]}"
      do
        for start_type in "${start_types[@]}"
        do
          for alpha in "${alphas[@]}"
          do
            for seed in {1..1}
            do
              echo "sbatch run_bpi_gvfs.sh $alpha $seed $start_type $type_behavior_policy $num_ep $exploration_steps $exp_next_val"
              sbatch -J $jobname ./run_bpi_gvfs.sh $alpha $seed $start_type $type_behavior_policy $num_ep $exploration_steps $exp_next_val
            done
          done
        done
      done
    done
  done
done