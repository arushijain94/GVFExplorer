#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=900MB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

temperature_steps=400
size=20
timeout=500
stochastic_env=1
n_target_pol=2
last_k=500

## Var (ours)
type_obj=1
beh_min=1e-3
exp_next_val=1
random_start_beh=2
type_behavior_policy=0
num_ep=1
max_alpha_Q=1.0
max_alpha_var=1.0
exploration_steps=500
num_iter=10000
lr_decay_steps_Q=300000
lr_decay_steps_var=300000

temp_name="6march"
seed=$1
env_name=$2

group_name=LR_Var_${env_name}_${temp_name}

python -m src.lr_sens_bpi --max_alpha_Q ${max_alpha_Q} --max_alpha_var ${max_alpha_var} --lr_decay_steps_Q ${lr_decay_steps_Q} --lr_decay_steps_var ${lr_decay_steps_var} --env_name ${env_name} --exp_next_val ${exp_next_val} --temperature_steps ${temperature_steps} --type_behavior_policy ${type_behavior_policy} --last_k ${last_k} --type_obj ${type_obj} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --random_start_beh ${random_start_beh} --size ${size} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --seed ${seed}
