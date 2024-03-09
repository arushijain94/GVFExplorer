#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/%j-%x.out

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad

group_name=BPI_multR_greedy_pol_blunder
num_iter=15000
temperature_steps=400
size=5
env_name=semigreedy_online_rooms_different_reward_grid
tar_pol_loc="/home/mila/j/jainarus/scratch/VarReduction/Det_10/ideal_targets"
timeout=100
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/Det_5/RLCResults'
stochastic_env=1
n_target_pol=2
last_k=20

## BPI (ours)

alpha=$1
seed=$2
random_start_beh=$3
type_behavior_policy=$4
name='BPI_m_greedy'
beh_min=1e-3
logs_name=${name}_alpha${alpha}_behstart${random_start_beh}_type_pol${type_behavior_policy}
wandb_name=${logs_name}
type_obj=1
num_ep=$5
exploration_steps=$6
exp_next_val=$7
python -m src.multiple_BPI --env_name ${env_name} --exp_next_val ${exp_next_val} --temperature_steps ${temperature_steps} --type_behavior_policy ${type_behavior_policy} --last_k ${last_k} --type_obj ${type_obj} --exploration_steps ${exploration_steps} --num_episodes ${num_ep} --random_start_beh ${random_start_beh} --seed ${seed} --size ${size} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPI/${logs_name}/seed_${seed}

# target TD
#num_ep=1
#num_iter=500
#alpha=0.1
#epsilon=0
#seed=1
#type_target=4 # Rb, Mixture
#name='Target_TD_try'
#wandb_name=${name}
#python -m src.multiple_target_td --env_name ${env_name} --seed ${seed} --tar_pol_loc ${tar_pol_loc} --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_TD/${wandb_name}/seed_${seed}



## Target MC
#num_ep=1
#num_iter=10000
#alpha=$1
#epsilon=$2
#seed=$3
#name='Target_MC_1'
#wandb_name=${name}
#python -m src.target_case --size ${size} --seed ${seed} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_MC/${wandb_name}/seed_${seed}

## Target TD
#num_ep=1
#num_iter=10000
#alpha=$1
#epsilon=$2
#seed=$3
#type_target=$4 # Rb, Mixture
#name='TD_new_5'
#wandb_name=${name}
#python -m src.multiple_target_td --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_TD/${wandb_name}/seed_${seed}
#

#
#
#num_ep=1
#num_iter=5000
#alpha=$1
#rho_min=0.01
#seed=$3
#K=$4
#epsilon=$2
#name='BPG_fin'
#wandb_name=${name}
#logs_name=${name}_alpha${alpha}_K${K}_rmin${rho_min}
#echo "python -m src.bpg --K ${K} --epsilon ${epsilon} --wandb_group $group_name --wandb_name ${wandb_name} --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPG/${logs_name}/seed_${seed}"
#python -m src.bpg --size ${size} --stochastic_env ${stochastic_env} --rho_min ${rho_min} --K ${K} --epsilon ${epsilon} --wandb_group $group_name --wandb_name ${wandb_name} --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPG/${logs_name}/seed_${seed}

