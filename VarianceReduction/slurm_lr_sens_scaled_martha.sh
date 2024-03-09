#!/bin/bash

jobname="LR_martha"

for seed in {1..10}
do
  sbatch -J $jobname ./run_lr_sens_scaled_martha.sh $seed
done