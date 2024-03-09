# Run multiple experiments on a single machine.
import subprocess
import time
import os

N_SEEDS = 50

learning_rates=[1e-3]
target_network_frequencys=[100]
prioritized_replays= [0, 1]
batch_sizes=[64]
train_frequency=4
learning_starts=200
exploration_step = 4000

buffer_sizes = [25000]
tau=1.0

for buffer_size in buffer_sizes:
    for batch_size in batch_sizes:
        for prioritized_replay in prioritized_replays:
            for learning_rate in learning_rates:
                for target_network_frequency in target_network_frequencys:
                    for seed in range(1, N_SEEDS+1):
                        args = [
                            "--learning-starts",
                            learning_starts,
                            "--train-frequency",
                            train_frequency,
                            "--batch-size",
                            batch_size,
                            "--buffer-size",
                            buffer_size,
                            "--target-network-frequency",
                            target_network_frequency,
                            "--learning-rate",
                            learning_rate,
                            "--prioritized-replay",
                            prioritized_replay,
                            "--exploration-step",
                            exploration_step,
                            "--tau",
                            tau,
                            "--seed",
                            seed
                        ]
                        args = list(map(str, args))
                        command = " ".join(
                            ["python", "-m", "src.martha_Q_rooms", "--config ./hyperparams/martha_Q_room_eval.yml", *args])
                        print(command)
                        # # call sbatch script
                        ok = subprocess.call(["sbatch", "--job-name=marthaQ", "pe_torchy.sh", command])
                        time.sleep(0.1)

print("All hyperparameter sweeps submitted.")