#!/bin/bash
#SBATCH --time=1:40:00
#SBATCH --mem=900MB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

size=20
timeout=500
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/Det_5/RLCResults'
stochastic_env=1
n_target_pol=2
last_k=500
num_ep=1
lr_decay_steps=300000
max_alpha=1.0
temp_name="Sens_1"
exploration_steps=500
num_iter=10000

seed=$1
env_name=$2
group_name=LR_TD_${env_name}_${temp_name}
python -m src.lr_sens_td --max_alpha ${max_alpha} --lr_decay_steps ${lr_decay_steps} --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --n_target_pol ${n_target_pol} --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout}  --seed ${seed}