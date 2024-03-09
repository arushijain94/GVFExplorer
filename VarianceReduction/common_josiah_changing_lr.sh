#!/bin/bash
#SBATCH --time=2:30:00
#SBATCH --mem=2G
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
max_alpha=1.0
lr_decay_steps=500000

min_alpha=$1
seed=$2
num_iter=$3
temp_name=$4
env_name=$5
exploration_steps=$6
num_ep=$7

group_name=Josiah_${env_name}_${temp_name}

name='Jos'
logs_name=${name}_alpha${min_alpha}

python -m src.common_josiah_changing_lr --min_alpha ${min_alpha} --max_alpha ${max_alpha} --lr_decay_steps ${lr_decay_steps} --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --seed ${seed} --size ${size} --n_target_pol ${n_target_pol} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Josiah/${logs_name}/seed_${seed}
