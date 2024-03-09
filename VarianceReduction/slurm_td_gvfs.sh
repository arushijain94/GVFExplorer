#!/bin/bash

type_tars=(0 1 2)
alphas=(0.25)
exploration_steps_list=(4000)
jobname='tdgvf'

for type_tar in "${type_tars[@]}"
do
  for alpha in "${alphas[@]}"
  do
      for exploration_steps in "${exploration_steps_list[@]}"
      do
        for seed in {1..30}
        do
          echo "sbatch run_td_gvfs.sh $alpha $exploration_steps $seed $type_tar"
          sbatch -J $jobname ./run_td_gvfs.sh $alpha $exploration_steps $seed $type_tar
        done
      done
  done
done