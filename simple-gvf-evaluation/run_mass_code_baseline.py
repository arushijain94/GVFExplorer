# Run multiple experiments on a single machine.
import subprocess
import time
import os

N_SEEDS = 5

learning_rates=[5e-4]
target_network_frequencys=[100]
prioritized_replays= [0, 1]
batch_size=64
train_frequency=4
learning_starts=200
exploration_steps = [4000]

# buffer_sizes = [1000, 5000] #25000
buffer_sizes = [25000] #25000
tau=1.0
type_target_policys=[1]

for type_target_policy in type_target_policys:
    for buffer_size in buffer_sizes:
        for exploration_step in exploration_steps:
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
                                seed,
                                "--type-target-policy",
                                type_target_policy
                            ]
                            args = list(map(str, args))
                            command = " ".join(
                                ["python", "-m", "src.baseline_policy_evaluation_rooms", "--config ./hyperparams/pol_eval_rooms.yml", *args])
                            print(command)
                            # # call sbatch script
                            ok = subprocess.call(["sbatch", "--job-name=base", "pe_torchy.sh", command])
                            time.sleep(0.05)

print("All hyperparameter sweeps submitted.")