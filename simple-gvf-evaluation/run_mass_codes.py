# Run multiple experiments on a single machine.
import subprocess
import time
import os

N_SEEDS = 5

learning_rates=[5e-3]
taus= [1.0]
q_type_loss= 1
var_type_loss=1
batch_size=64
train_frequency=4
learning_starts=200
target_network_frequency=100
variance_train_frequencys=[2]

# buffer_size = 25000 (original size)
# buffer_sizes=[1000, 5000] # abalation study
buffer_sizes=[25000]
type_next_target_vals=[0]
behavior_policy_types=[1]
exploration_steps = [4000]
temperature_steps=[50000]
prioritized_replays=[1, 0]
beta_fractions=[0.6]
prioritized_replay_alphas=[0.8]
start_ts=[1.]
end_ts=[0.1]
start_t=1.0

for buffer_size in buffer_sizes:
    for end_t in end_ts:
        for beta_fraction in beta_fractions:
            for behavior_policy_type in behavior_policy_types:
                for variance_train_frequency in variance_train_frequencys:
                    for prioritized_replay in prioritized_replays:
                        for temperature_step in temperature_steps:
                            for learning_rate in learning_rates:
                                for exploration_step in exploration_steps:
                                    for prioritized_replay_alpha in prioritized_replay_alphas:
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
                                                "--exploration-step",
                                                exploration_step,
                                                "--temperature-step",
                                                temperature_step,
                                                "--variance-train-frequency",
                                                variance_train_frequency,
                                                "--learning-rate",
                                                learning_rate,
                                                "--var-type-loss",
                                                var_type_loss,
                                                "--prioritized-replay",
                                                prioritized_replay,
                                                "--prioritized-replay-alpha",
                                                prioritized_replay_alpha,
                                                "--beta-fraction",
                                                beta_fraction,
                                                "--q-type-loss",
                                                q_type_loss,
                                                "--type-next-target-val",
                                                0,
                                                "--behavior-policy-type",
                                                behavior_policy_type,
                                                "--start-t",
                                                start_t,
                                                "--end-t",
                                                end_t,
                                                "--seed",
                                                seed
                                            ]
                                            args = list(map(str, args))
                                            command = " ".join(
                                                ["python", "-m", "src.variance_based_sampling_rooms", "--config ./hyperparams/var_based_eval_room.yml", *args])
                                            print(command)
                                            # # call sbatch script
                                            ok = subprocess.call(["sbatch", "--job-name=var", "pe_torchy.sh", command])
                                            time.sleep(0.025)

print("All hyperparameter sweeps submitted.")