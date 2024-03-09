import argparse
import envs
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
from src.common import load_config, linear_schedule, make_env, create_env_args
from src.multi_dim_dqn import DQNAgent, parse_args
import warnings
import json
import subprocess
import shlex
from src.simple_target_policy import SimpleTargetPolicy
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


class StateCounter:
    def __init__(self, env, bins=30, proportion_of_states=0.3):
        self.env = env
        self.bins = bins
        self.state_counts = {}
        self.discrete_to_continuous_mapping = {}
        self.proportion_of_states = proportion_of_states
        self.bins_per_dimension = [np.linspace(low, high, num=self.bins) for low, high in
                                   zip(self.env.observation_space.low, self.env.observation_space.high)]

    def discretize_state(self, state):
        discrete_state = tuple(np.digitize(s, b) for s, b in zip(state, self.bins_per_dimension))
        return discrete_state


    def update_counts(self, state):
        disc_state = self.discretize_state(state)
        if disc_state in self.state_counts:
            self.state_counts[disc_state] += 1
            self.discrete_to_continuous_mapping[disc_state].append(state)
        else:
            self.state_counts[disc_state] = 1
            self.discrete_to_continuous_mapping[disc_state] = [state]

    def get_most_common_states(self, n):
        sorted_states = sorted(self.state_counts.keys(), key=lambda x: self.state_counts[x], reverse=True)
        # print("sorted states:", self.state_counts)
        most_common_states = []
        count = 0

        for disc_state in sorted_states:
            if count >= n:
                break
            available_states = np.array(self.discrete_to_continuous_mapping[disc_state])
            proportion = min(int(len(available_states)*self.proportion_of_states), n - count)
            # print("len of states:", len(available_states))
            selected_indices = np.random.choice(len(available_states), proportion, replace=False)
            selected_states = available_states[selected_indices]
            # print("selected states:", selected_states)
            most_common_states.extend(selected_states)
            # print("count:", count)
            count += proportion

        return most_common_states

def get_state_dist_acc_to_agent(agent, env, num_episodes=2, num_states=5):
    # given a policy, computes the most commonly occurring states acc. to policy distribution
    counter = StateCounter(env, proportion_of_states=0.1)
    for eps in range(num_episodes):
        eps_reward = 0.
        obs, _ = env.reset()
        done = False
        max_steps = 200
        curr_step = 0
        while not done and curr_step<=max_steps:
            actions = agent.select_action(obs, epsilon=0, greedy=False)
            next_obs, rewards, terminations, truncations, _ = env.step(actions[0])
            done = terminations or truncations
            counter.update_counts(obs)
            obs = next_obs
            eps_reward += np.mean(rewards)
            curr_step+=1
    most_common_states = counter.get_most_common_states(n=num_states)
    return np.array(most_common_states)

def get_random_start_states(env, num_states=5):
    unique_states = set()
    while len(unique_states)< num_states:
        obs, _ = env.reset()
        unique_states.add(tuple(obs))
    return np.array(list(unique_states))

def get_strategic_start_states(env, env_id, num_samples=1, bins=20):
    start_states = []
    bin_size = 1./bins
    total_retry = 5
    for i in range(0, bins):
        for j in range(0, bins):
            retry=0
            start_x = i*bin_size
            start_y = j*bin_size
            while retry < total_retry:
                retry += 1
                data = np.random.uniform((start_x,start_y), (start_x+bin_size, start_y+bin_size), (num_samples, 2))
                data = np.clip(data, a_min=0., a_max=1.)
                if env_id == "FourRoomsEnv-v0":
                    if env.is_legal_start_state(data):
                        start_states.extend(data)
                        break
                else:
                    start_states.extend(data)
                    break
            # print(f"retry {retry}, start state {data}")
    start_states = np.array(start_states)
    start_states = np.round(start_states, 4)
    # print(f"start state: {start_states}, len: {start_states.shape}")
    return start_states

def submit_sbatch_job(initial_state):
    # Convert numpy array to string format with square brackets
    initial_state_str = " ".join(map(str, initial_state))

    # print("state in compute_start_state:", initial_state_str)
    command = " ".join(["python", "-m", "src.true_policy_evaluation_rooms", "--start-state", initial_state_str,
                        "--config", "./hyperparams/FR_goal1.yml"])
    print("command:", command)
    # # call sbatch script
    ok = subprocess.call(["sbatch", "mc_torchy3.sh", command])
    time.sleep(0.01)


if __name__ == "__main__":
    args = parse_args()

    env_args = create_env_args(args, args.env_id)
    env = make_env(args.env_id, args.seed, args.capture_video, run_name=None, env_args=env_args)  # training

    # get random states:
    random_states = get_strategic_start_states(env, args.env_id)
    #
    for state in random_states:
        submit_sbatch_job(state)