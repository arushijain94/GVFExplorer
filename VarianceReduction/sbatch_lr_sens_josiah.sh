#!/bin/bash
#SBATCH --time=1:00:00
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
n_target_pol=2
last_k=500
max_alpha=1.0
lr_decay_steps=300000

min_alpha=0.1
temp_name="Sens_1"
exploration_steps=500
num_iter=10000
num_ep=1

seed=$1
env_name=$2
#env_name=online_rooms_same_reward_grid
group_name=LR_Josiah_${env_name}_${temp_name}

name='Jos'
logs_name=${name}_alpha${min_alpha}

python -m src.lr_sens_josiah --min_alpha ${min_alpha} --max_alpha ${max_alpha} --lr_decay_steps ${lr_decay_steps} --env_name ${env_name} --last_k ${last_k} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --seed ${seed} --size ${size} --n_target_pol ${n_target_pol} --stochastic_env ${stochastic_env} --wandb_group $group_name --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Josiah/${logs_name}/seed_${seed}
