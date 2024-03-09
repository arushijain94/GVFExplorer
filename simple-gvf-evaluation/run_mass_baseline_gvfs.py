# Run multiple experiments on a single machine.
import subprocess
import time
import os

N_SEEDS = 50

learning_rates=[1e-3]
prioritized_replays= [0, 1]
exploration_steps = [4000]
type_target_policys = [1, 2]

for type_target_policy in type_target_policys:
    for exploration_step in exploration_steps:
        for prioritized_replay in prioritized_replays:
            for learning_rate in learning_rates:
                for seed in range(1, N_SEEDS+1):
                    args = [
                        "--learning-rate",
                        learning_rate,
                        "--prioritized-replay",
                        prioritized_replay,
                        "--exploration-step",
                        exploration_step,
                        "--seed",
                        seed,
                        "--type-target-policy",
                        type_target_policy
                    ]
                    args = list(map(str, args))
                    command = " ".join(
                        ["python", "-m", "src.baseline_gvfs", "--config ./hyperparams/baseline_gvf_eval.yml", *args])
                    print(command)
                    # # call sbatch script
                    ok = subprocess.call(["sbatch", "--job-name=baseGVF", "pe_torchy.sh", command])
                    time.sleep(0.05)

print("All hyperparameter sweeps submitted.")