#!/bin/bash

Ks=(0)
alphas=(1e-4 1e-3 1e-2 1e-1)
epsilons=(0.1)
jobname='bpi'

for k in "${Ks[@]}"
do
  for alpha in "${alphas[@]}"
  do
      for epsilon in "${epsilons[@]}"
      do
        for seed in {1..1}
        do
          echo "sbatch run.sh $alpha $epsilon $seed $k"
          sbatch -J $jobname ./run_mlt.sh $alpha $epsilon $seed $k
        done
      done
  done
done