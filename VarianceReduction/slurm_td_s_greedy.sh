#!/bin/bash

type_tars=(0)
alphas=(0.25)
exploration_steps_list=(1000)
jobname='td_greedy'

for type_tar in "${type_tars[@]}"
do
  for alpha in "${alphas[@]}"
  do
      for exploration_steps in "${exploration_steps_list[@]}"
      do
        for seed in {1..50}
        do
          echo "sbatch run_td_s_greedy.sh $alpha $exploration_steps $seed $type_tar"
          sbatch -J $jobname ./run_td_s_greedy.sh $alpha $exploration_steps $seed $type_tar
        done
      done
  done
done