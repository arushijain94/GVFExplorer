#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

#group_name=BPI_online_single_reward_grid_26Feb
#num_iter=10000
temperature_steps=400
size=5
#env_name=online_rooms_same_reward_grid\
timeout=100
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/Det_5/RLCResults'
stochastic_env=1
n_target_pol=2
last_k=20

## Var (ours)
type_obj=1
beh_min=1e-3
exp_next_val=1
random_start_beh=2
type_behavior_policy=0
num_ep=1

alpha=$1
seed=$2
num_iter=$3
temp_name=$4
env_name=$5
exploration_steps=$6

group_name=Var_${env_name}_${temp_name}

name='BPI'
logs_name=${name}_alpha${alpha}_type_pol${type_behavior_policy}

python -m src.multiple_BPI --env_name ${env_name} --exp_next_val ${exp_next_val} --temperature_steps ${temperature_steps} --type_behavior_policy ${type_behavior_policy} --last_k ${last_k} --type_obj ${type_obj} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --random_start_beh ${random_start_beh} --seed ${seed} --size ${size} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPI/${logs_name}/seed_${seed}
