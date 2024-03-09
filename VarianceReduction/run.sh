#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_out/%j-%x.out
#SBATCH --cpus-per-task=1

module load anaconda/3
source ~/.bashrc
conda activate behpolgrad
group_name=GW_Multiple
size=10
timeout=300
logs_base_dir='/home/mila/j/jainarus/scratch/VarReduction/GW_10/MultipleTargets'
stochastic_env=0
n_target_pol=3
## generate target policies
#num_ep=10
#alpha=0.1
#name='tar_v1'
#wandb_name=${name}
#logs_name=${name}_alpha${alpha}
#size=5
#python -m src.control_td_learning --size ${size} --stochastic_env ${stochastic_env} --num_episodes ${num_ep} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/Control/${logs_name}



## BPI (ours)
num_ep=1
num_iter=30000
alpha=$1
epsilon=$2
seed=$3
name='Mult_BPI_4'
beh_min=$4
wandb_name=${name}
logs_name=${name}_alpha${alpha}_eps${epsilon}
python -m src.multiple_BPI --size ${size} --n_target_pol ${n_target_pol} --beh_min ${beh_min} --rho_min ${beh_min} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --epsilon ${epsilon} --num_episodes ${num_ep} --num_iter ${num_iter} --timeout ${timeout} --alpha ${alpha} --logdir ${logs_base_dir}/BPI/${logs_name}/seed_${seed}

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
#num_ep=1
#num_iter=50000
#alpha=$1
#epsilon=0.
#seed=$3
#name='Target_TD_3'
#wandb_name=${name}
#python -m src.target_td --size ${size} --stochastic_env ${stochastic_env} --wandb_group $group_name --wandb_name $wandb_name --num_episodes ${num_ep} --alpha ${alpha} --num_iter ${num_iter} --timeout ${timeout} --logdir ${logs_base_dir}/Target_TD/${wandb_name}/seed_${seed}


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

