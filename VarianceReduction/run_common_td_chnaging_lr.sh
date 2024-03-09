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


## Target TD
num_ep=1
max_alpha=1.0

min_alpha=$1
exploration_steps=$2
seed=$3
type_target=$4 # Rb, Mixture
env_name=$5
temp_name=$6
num_iter=$7
lr_decay_steps=$8

group_name=TD_${env_name}_${temp_name}
logs_name=TD_${env_name}_alpha${min_alpha}_type_tar${type_target}

python -m src.common_td_changing_lr --max_alpha ${max_alpha} --min_alpha ${min_alpha} --lr_decay_steps ${lr_decay_steps} --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/TD_SingeReward/${logs_name}/seed_${seed}