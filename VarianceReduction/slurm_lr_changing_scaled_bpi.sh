#!/bin/bash

jobname="LR_BPI"

for seed in {1..10}
do
  sbatch -J $jobname ./run_lr_changing_scaled_bpi.sh $seed
done
