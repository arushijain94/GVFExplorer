#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time=03:00:00  # 1 hr
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%x-%j.out

echo "In sbatch script!"
echo "Working Directory: $(pwd)"


module load anaconda/3
conda activate rl-baselines3-zoo

python -m src.variance_duelling_sampling_rooms --config ./hyperparams/var_room_duel.yml