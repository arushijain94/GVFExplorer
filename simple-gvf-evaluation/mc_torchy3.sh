#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time=10:00:00  # 1 hr
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%x-%j.out

echo "In sbatch script!"
echo "Working Directory: $(pwd)"
command_to_exec="$1"

module load anaconda/3
conda activate rl-baselines3-zoo

echo "$command_to_exec"
eval "$command_to_exec"