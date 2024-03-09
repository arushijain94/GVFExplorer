#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_out/%j-%x.out
#SBATCH --cpus-per-task=1

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad
group_name=tmaze
env_name=tmaze
timeout=300
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/tmaze/MultipleTargets'
stochastic_env=0
n_target_pol=4
tar_pol_loc='/home/mila/j/jainarus/scratch/VarReduction/tmaze/ideal_targets'

## generate target policies
#num_ep=10
#alpha=0.1
#name='tar_v1'
#wandb_name=${name}
#logs_name=${name}_alpha${alpha}
#size=5
#python -m src.control_td_learning --size ${size} --stochastic_env ${stochastic_env} --num_episodes ${num_ep} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/Control/${logs_name}



## BPI (ours)
#num_ep=1
#num_iter=7000
#alpha=$1
#epsilon=$2
#seed=$3
#name='BPI_12'
#beh_min=0.01
#random_start_beh=$4
#type_obj=$5
#last_k=$6
#wandb_name=${name}
#logs_name=${name}_alpha${alpha}_eps${epsilon}
#python -m src.multiple_BPI --last_k ${last_k} --type_obj ${type_obj} --random_start_beh ${random_start_beh} --seed ${seed} --env_name ${env_name} --tar_pol_loc ${tar_pol_loc} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --epsilon ${epsilon} --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPI/${logs_name}/seed_${seed}


#num_ep=1
#num_iter=100
#alpha=0.9
#epsilon=0.1
#seed=1
#name='try'
#beh_min=0.01
#random_start_beh=1
#type_obj=4
#wandb_name=${name}
#logs_name=${name}_alpha${alpha}_eps${epsilon}
#python -m src.multiple_BPI --type_obj ${type_obj} --random_start_beh ${random_start_beh} --seed ${seed} --env_name ${env_name} --tar_pol_loc ${tar_pol_loc} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --epsilon ${epsilon} --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPI/${logs_name}/seed_${seed}


## Target MC
#num_ep=1
#num_iter=10000
#alpha=$1
#epsilon=$2
#seed=$3
#name='Target_MC_1'
#wandb_name=${name}
#python -m src.target_case --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_MC/${wandb_name}/seed_${seed}

## Target TD
num_ep=1
num_iter=5000
alpha=$1
epsilon=$2
seed=$3
type_target=$4 # Rb, Mixture
name='Target_TD_12'
wandb_name=${name}
python -m src.multiple_target_td --env_name ${env_name} --seed ${seed} --tar_pol_loc ${tar_pol_loc} --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_TD/${wandb_name}/seed_${seed}

#
#num_ep=1
#num_iter=500
#alpha=0.1
#epsilon=1.
#seed=1
#type_target=0 # Rb, Mixture
#name='try_1'
#wandb_name=${name}
#python -m src.multiple_target_td --env_name ${env_name} --seed ${seed} --tar_pol_loc ${tar_pol_loc} --n_target_pol ${n_target_pol} --seed ${seed} --type_target ${type_target} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_TD/${wandb_name}/seed_${seed}


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

