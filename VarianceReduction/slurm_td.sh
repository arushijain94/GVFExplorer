#!/bin/bash

type_tars=(1 2)
alphas=(0.8)
exploration_steps_list=(4000)
jobname='td'

for type_tar in "${type_tars[@]}"
do
  for alpha in "${alphas[@]}"
  do
      for exploration_steps in "${exploration_steps_list[@]}"
      do
        for seed in {1..50}
        do
          echo "sbatch run_td.sh $alpha $exploration_steps $seed $type_tar"
          sbatch -J $jobname ./run_td.sh $alpha $exploration_steps $seed $type_tar
        done
      done
  done
done