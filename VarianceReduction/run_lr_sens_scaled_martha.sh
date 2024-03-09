#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH --mem=1G
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
type_behavior_policy=1

## BPI (ours)
beh_min=1e-3
random_start_beh=2
num_ep=1
type_obj=1
exp_next_val=1
behv_reward_combined=1


min_alpha=0.1
temp_name="lr_changing"
num_iter=40000

seed=$1

group_name=SRScaled_${temp_name}
name='SRScale'

logs_name=${name}_type_pol${type_behavior_policy}

python -m src.lr_sens_martha_scaled --max_alpha ${max_alpha} --min_alpha ${min_alpha} --lr_decay_steps ${lr_decay_steps} --n_target_pol ${n_target_pol} --num_reward_obj ${num_reward_obj} --size ${size} --behv_reward_combined ${behv_reward_combined} --env_name ${env_name} --exp_next_val ${exp_next_val} --type_behavior_policy ${type_behavior_policy} --last_k ${last_k} --type_obj ${type_obj} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --random_start_beh ${random_start_beh} --seed ${seed} --beh_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/SR/${logs_name}/seed_${seed}

