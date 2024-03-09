#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=2G
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

temperature_steps=400
env_name=grid_scaled_up_exp
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/Det_5/RLCResults'
stochastic_env=1
last_k=500
max_alpha_Q=1.0
max_alpha_var=1.0
exploration_steps=500
lr_decay_steps_Q=500000
lr_decay_steps_var=$lr_decay_steps_Q
random_start_beh=2
num_ep=1
timeout=500
n_target_pol=4
num_reward_obj=10
size=20

## BPI (ours)
name='BPI'
beh_min=1e-3
type_behavior_policy=0
type_obj=1
exp_next_val=1

min_alpha_Q=$1
seed=$2
random_start_beh=$3
temp_name=$4
min_alpha_var=$5
num_iter=$6

group_name=VarScaled_${temp_name}_10Reward_New
name='VarScale'
logs_name=${name}_behstart${random_start_beh}

python -m src.multiple_BPI_scaled_exp --size ${size} --n_target_pol ${n_target_pol} --num_reward_obj ${num_reward_obj} --max_alpha_Q ${max_alpha_Q} --max_alpha_var ${max_alpha_var} --min_alpha_var ${min_alpha_var} --min_alpha_Q ${min_alpha_Q} --lr_decay_steps_Q ${lr_decay_steps_Q} --lr_decay_steps_var ${lr_decay_steps_var} --env_name ${env_name} --exp_next_val ${exp_next_val} --temperature_steps ${temperature_steps} --type_behavior_policy ${type_behavior_policy} --last_k ${last_k} --type_obj ${type_obj} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --random_start_beh ${random_start_beh} --seed ${seed} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/BPI/${logs_name}/seed_${seed}
