#!/bin/bash

jobname="LR_TD"
env_name='semigreedy_online_rooms_different_reward_grid'

for seed in {1..10}
do
  sbatch -J $jobname ./sbatch_lr_sen_td.sh $seed $env_name
done