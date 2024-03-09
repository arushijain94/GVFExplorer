# Run multiple experiments on a single machine.
import subprocess
import time
import os

N_SEEDS = 10

learning_rates=[5e-3]
prioritized_replays=[0, 1]

behavior_policy_types=[1]
exploration_steps = [4000]
temperature_steps=[50000]
beta_fractions=[0.2]
prioritized_replay_alphas=[0.6]
start_ts=[1.]
end_ts=[0.1]

for start_t in start_ts:
    for end_t in end_ts:
        for beta_fraction in beta_fractions:
            for behavior_policy_type in behavior_policy_types:
                for prioritized_replay in prioritized_replays:
                    for temperature_step in temperature_steps:
                        for learning_rate in learning_rates:
                            for exploration_step in exploration_steps:
                                for prioritized_replay_alpha in prioritized_replay_alphas:
                                    for seed in range(1, N_SEEDS+1):
                                        args = [
                                            "--exploration-step",
                                            exploration_step,
                                            "--temperature-step",
                                            temperature_step,
                                            "--learning-rate",
                                            learning_rate,
                                            "--prioritized-replay",
                                            prioritized_replay,
                                            "--prioritized-replay-alpha",
                                            prioritized_replay_alpha,
                                            "--beta-fraction",
                                            beta_fraction,
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
                                            ["python", "-m", "src.variance_gvfs", "--config ./hyperparams/var_gvf_eval.yml", *args])
                                        print(command)
                                        # # call sbatch script
                                        ok = subprocess.call(["sbatch", "--job-name=varGVF", "pe_torchy.sh", command])
                                        time.sleep(0.05)

print("All hyperparameter sweeps submitted.")