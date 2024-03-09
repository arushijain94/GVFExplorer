#!/bin/bash
#SBATCH --time=4:30:00
#SBATCH --mem=900MB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

size=5
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
env_name=$5
temp_name=$6
num_iter=$7

group_name=TD_${env_name}_${temp_name}
name='TD'
logs_name=${name}_alpha${alpha}_targettype${type_target}

python -m src.multiple_target_td --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/TD_SingeReward/${logs_name}/seed_${seed}