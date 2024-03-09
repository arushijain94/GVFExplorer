#!/bin/bash

jobname="LR_TD"

for seed in {1..10}
do
  sbatch -J $jobname ./run_lr_chnaging_scaled_td.sh $seed
done
