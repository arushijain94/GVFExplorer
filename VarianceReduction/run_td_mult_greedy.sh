#!/bin/bash
#SBATCH --time=6:30:00
#SBATCH --mem=900MB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

group_name=TD_grid_multR_greedy_pol_blunder
num_iter=15000
size=5
env_name=semigreedy_online_rooms_different_reward_grid
tar_pol_loc="/home/mila/j/jainarus/scratch/VarReduction/Det_10/ideal_targets"
timeout=100
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/Det_5/RLCResults'
stochastic_env=1
n_target_pol=2
last_k=20


## Target TD
num_ep=1
alpha=$1
exploration_steps=$2
seed=$3
type_target=$4 # Rb, Mixture
name='TD_m_greedy'
logs_name=${name}_alpha${alpha}_targettype${type_target}
wandb_name=${logs_name}
python -m src.multiple_target_td --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/TD_SingeReward/${wandb_name}/seed_${seed}