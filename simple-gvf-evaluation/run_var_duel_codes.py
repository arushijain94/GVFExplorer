# Run multiple experiments on a single machine.
import subprocess
import time
import os

N_SEEDS = 5

learning_rates=[1e-4, 1e-3, 5e-3]
target_network_frequencys=[100]
taus= [1.0]
train_frequencys= [4]
q_type_losses= [1]
var_type_losses=[1]

buffer_sizes = [25000, 50000]
variance_train_frequency=2

for buffer_size in buffer_sizes:
    for tau in taus:
        for q_type_loss in q_type_losses:
            for var_type_loss in var_type_losses:
                for train_frequency in train_frequencys:
                    for learning_rate in learning_rates:
                        for target_network_frequency in target_network_frequencys:
                            for seed in range(1, N_SEEDS+1):
                                args = [
                                    "--buffer-size",
                                    buffer_size,
                                    "--target-network-frequency",
                                    target_network_frequency,
                                    "--train-frequency",
                                    train_frequency,
                                    "--learning-rate",
                                    learning_rate,
                                    "--variance-train-frequency",
                                    variance_train_frequency,
                                    "--var-type-loss",
                                    var_type_loss,
                                    "--q-type-loss",
                                    q_type_loss,
                                    "--tau",
                                    tau,
                                    "--seed",
                                    seed
                                ]
                                args = list(map(str, args))
                                command = " ".join(
                                    ["python", "-m", "src.variance_duelling_sampling_rooms", "--config ./hyperparams/var_based_eval_room.yml", *args])
                                print(command)
                                # # call sbatch script
                                ok = subprocess.call(["sbatch", "pe_torchy.sh", command])
                                time.sleep(0.05)

print("All hyperparameter sweeps submitted.")