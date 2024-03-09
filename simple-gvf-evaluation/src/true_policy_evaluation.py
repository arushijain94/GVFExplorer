import argparse
import envs
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
from src.common import load_config, linear_schedule, make_env
from src.multi_dim_dqn import DQNAgent, parse_args
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


# def parse_args():
#     # fmt: off
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, help="path to config file", default=None)
#     parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
#         help="the name of this experiment")
#     parser.add_argument("--seed", type=int, default=1,
#         help="seed of the experiment")
#     parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="if toggled, `torch.backends.cudnn.deterministic=False`")
#     parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="if toggled, cuda will be enabled by default")
#     parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="if toggled, this experiment will be tracked with Weights and Biases")
#     parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
#         help="the wandb's project name")
#     parser.add_argument("--wandb-entity", type=str, default=None,
#         help="the entity (team) of wandb's project")
#     parser.add_argument("--wandb-group", type=str, default=None,
#                         help="the group (team) of wandb's project")
#     parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="whether to capture videos of the agent performances (check out `videos` folder)")
#     parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="whether to save model into the `runs/{run_name}` folder")
#     parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
#         help="whether to upload the saved model to huggingface")
#     parser.add_argument("--hf-entity", type=str, default="",
#         help="the user or org name of the model repository from the Hugging Face Hub")
#     parser.add_argument("--run-save-loc", type=str, default="",
#                         help="save location for the data")
#
#
#     # Algorithm specific arguments
#     parser.add_argument("--env-id", type=str, default="CartPole-v1",
#         help="the id of the environment")
#     parser.add_argument("--goal-position", type=float, default=0.0,
#                         help="position of goal for pendulum np.pi")
#
#     parser.add_argument("--total-timesteps", type=int, default=500000,
#         help="total timesteps of the experiments")
#     parser.add_argument("--learning-rate", type=float, default=2.5e-4,
#         help="the learning rate of the optimizer")
#     parser.add_argument("--buffer-size", type=int, default=10000,
#         help="the replay memory buffer size")
#     parser.add_argument("--gamma", type=float, default=0.99,
#         help="the discount factor gamma")
#     parser.add_argument("--tau", type=float, default=1.,
#         help="the target network update rate")
#     parser.add_argument("--target-network-frequency", type=int, default=500,
#         help="the timesteps it takes to update the target network")
#     parser.add_argument("--batch-size", type=int, default=128,
#         help="the batch size of sample from the reply memory")
#     parser.add_argument("--start-e", type=float, default=1,
#         help="the starting epsilon for exploration")
#     parser.add_argument("--end-e", type=float, default=0.05,
#         help="the ending epsilon for exploration")
#     parser.add_argument("--exploration-fraction", type=float, default=0.5,
#         help="the fraction of `total-timesteps` it takes from start-e to go end-e")
#     parser.add_argument("--learning-starts", type=int, default=10000,
#         help="timestep to start learning")
#     parser.add_argument("--train-frequency", type=int, default=10,
#         help="the frequency of training")
#     parser.add_argument("--eval-frequency", type=int, default=40,
#                         help="the frequency of evaluation")
#     parser.add_argument("--checkpoint-frequency", type=int, default=40,
#                         help="the frequency of evaluation")
#     parser.add_argument("--eval-episodes", type=int, default=5,
#                         help="the episodes of evaluation")
#     parser.add_argument(
#         "--evaluate-checkpoint", action="store_true", default=False, help="to evaluate a particular model"
#     )
#     parser.add_argument(
#         "--checkpoint-number", type=int, default=5000, help="to evaluate a checkpoint"
#     )

class MonteCarloValueEstimator:
    def __init__(self, env, agent, gamma=0.99, num_episodes=10):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.num_episodes = num_episodes # this is how many episode to roll out for each state


    def generate_episode(self, start_state):
        obs, _ = self.env.reset(options={"x":start_state[0], "y": start_state[1], "thetadot": start_state[2]})
        done = False
        eps_reward = 0.0
        while not done:
            actions = self.agent.select_action(obs, epsilon=0, greedy=False)
            next_obs, rewards, terminations, truncations, _ = self.env.step(actions[0])
            done = terminations or truncations
            obs = next_obs
            eps_reward += np.mean(rewards)
        return eps_reward


    def estimate_value_function(self, state):
        returns = []
        for _ in range(self.num_episodes):
            return_val = self.generate_episode(state)
            # print(f"eps {_}, return: {return_val}")
            returns.append(return_val)
        return np.mean(returns) # E[G] = V

    def estimate_value_function(self, states):
        value_function = {}
        for state in states:
            # print("state:", state)
            value = self.estimate_state_value(state)
            value_function[tuple(state)] = value
        return value_function


class StateCounter:
    def __init__(self, env, bins=[30,30,30]):
        self.env = env
        self.bins = bins
        self.state_counts = {}
        self.discrete_to_continuous_mapping = {}
        self.proportion_of_states = 0.3
        self.bins_per_dimension = [np.linspace(low, high, num=bin) for low, high, bin in
                                   zip(self.env.observation_space.low, self.env.observation_space.high,
                                       self.bins)]

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
            count += proportion

        return most_common_states

def get_state_dist_acc_to_agent(agent, env, num_episodes=2, num_states=5):
    # given a policy, computes the most commonly occuring states acc. to policy distribution
    counter = StateCounter(env)
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


def save_data(data_dict, save_loc):
    # Convert tuple keys to string
    state_to_val_str_keys = {str(k): v for k, v in data_dict.items()}

    with open(os.path.join(save_loc, 'state_to_val.json'), 'w') as f:
        json.dump(state_to_val_str_keys, f)


def load_data(load_loc):
    def convert_str_to_tuple(s):
        return tuple(map(float, s.strip("()").split(", ")))

    with open(os.path.join(load_loc, 'state_to_val.json'), 'r') as f:
        state_to_val_str_keys = json.load(f)

    state_to_val = {convert_str_to_tuple(k): v for k, v in state_to_val_str_keys.items()}
    return state_to_val


if __name__ == "__main__":
    args = parse_args()
    agent = DQNAgent(args)

    # load the agent from particular checkpoint
    agent.load_checkpoint(args.checkpoint_number)
    print("the agent is loaded form the checkpoint")

    env_args = {"goal_positions":[args.goal_position]}
    print("goal position:", args.goal_position)
    env = make_env(args.env_id, args.seed, args.capture_video, run_name=None, env_args=env_args)  # training

    # set agent into eval mode
    agent.q_network.eval()
    agent.target_network.eval()

    # Get the n most common continuous states
    most_common_states = get_state_dist_acc_to_agent(agent, env, num_episodes=2000, num_states=1000) # get [n * d] array

    # MC estimator
    mc = MonteCarloValueEstimator(env, agent, num_episodes=100000)
    state_to_value_dict = mc.estimate_value_function(most_common_states)

    # save data
    save_file_folder = "state_to_true_value"
    save_loc = os.path.join(args.run_save_loc, save_file_folder)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    save_data(state_to_value_dict, save_loc)

    # load data
    # load_loc = os.path.join(args.run_save_loc, save_file_folder)
    # loaded_state_val = load_data(load_loc)
    # print("loaded state val:", loaded_state_val)


    # close
    env.close()
    agent.env.close()

