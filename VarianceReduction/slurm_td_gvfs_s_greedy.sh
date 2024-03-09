#!/bin/bash

type_tars=(0)
alphas=(0.1 0.25 0.5 0.8 0.9)
epsilons=(0.1)
jobname='tdgvf_greedy'

for type_tar in "${type_tars[@]}"
do
  for alpha in "${alphas[@]}"
  do
      for epsilon in "${epsilons[@]}"
      do
        for seed in {1..5}
        do
          echo "sbatch run_td_gvfs.sh $alpha $epsilon $seed $type_tar"
          sbatch -J $jobname ./run_td_gvfs.sh $alpha $epsilon $seed $type_tar
        done
      done
  done
done