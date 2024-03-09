#!/bin/bash

type_tars=(0)
alphas=(0.8 0.25)
exploration_steps_list=(0.1)
jobname='td_m_greedy'

for type_tar in "${type_tars[@]}"
do
  for alpha in "${alphas[@]}"
  do
      for exploration_steps in "${exploration_steps_list[@]}"
      do
        for seed in {1..20}
        do
          echo "sbatch run_td_mult_greedy.sh $alpha $exploration_steps $seed $type_tar"
          sbatch -J $jobname ./run_td_mult_greedy.sh $alpha $exploration_steps $seed $type_tar
        done
      done
  done
done