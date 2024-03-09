#!/bin/bash
#SBATCH --time=3:20:00
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
n_target_pol=8
num_reward_obj=16
last_k=500
env_name=grid_scaled_up_exp
num_ep=1
max_alpha=1.0
exploration_steps=500
lr_decay_steps=500000
min_alpha=0.1
type_target=0
num_iter=20000

## Target TD
#min_alpha=$1
seed=$1
temp_name="3mar"

group_name=TDScaled_LRChanging_${temp_name}_NEW
name='TDScale'
logs_name=${name}_targettype${type_target}

python -m src.lr_sens_td_scale --n_target_pol ${n_target_pol} --num_reward_obj ${num_reward_obj} --size ${size} --lr_decay_steps ${lr_decay_steps} --max_alpha ${max_alpha} --min_alpha ${min_alpha} --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --seed ${seed} --type_target ${type_target} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/TD_Scaled/${logs_name}/seed_${seed}