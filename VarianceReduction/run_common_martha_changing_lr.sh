#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=2G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

temperature_steps=400
size=20
timeout=500
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/Det_5/RLCResults'
stochastic_env=1
n_target_pol=2
last_k=500
random_start_beh=2


## SR (Martha)
exp_next_val=1
beh_min=1e-3
type_obj=1
num_ep=1
max_alpha=1.0


min_alpha=$1
seed=$2
env_name=$3
type_behavior_policy=$4
name='SR'
logs_name=${name}_alpha${min_alpha}
temp_name=$5
exploration_steps=$6
behv_reward_combined=$7
num_iter=$8
lr_decay_steps=$9

group_name=SR_${env_name}_${temp_name}

python -m src.common_martha_chnaging_lr --lr_decay_steps ${lr_decay_steps} --max_alpha ${max_alpha} --behv_reward_combined ${behv_reward_combined} --env_name ${env_name} --exp_next_val ${exp_next_val} --temperature_steps ${temperature_steps} --type_behavior_policy ${type_behavior_policy} --last_k ${last_k} --type_obj ${type_obj} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --random_start_beh ${random_start_beh} --seed ${seed} --size ${size} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --min_alpha ${min_alpha} --logdir ${logs_base_dir}/SR/${logs_name}/seed_${seed}
