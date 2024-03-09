import argparse
import envs
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
from src.common import load_config, linear_schedule, make_env, create_env_args
from src.multi_dim_dqn import DQNAgent
import warnings
import json
from src.simple_target_policy import SimpleTargetPolicy

# import dask
# # import dask.bag as db
# # from dask.distributed import Client, progress, as_completed
# import dask
# from dask import delayed, compute
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", default=None)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="the group (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--run-save-loc", type=str, default="",
                        help="save location for the data")


    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--goal-position", type=float, default=0.0,
                        help="position of goal for pendulum np.pi")
    # takes argument of any number of size with at least one
    parser.add_argument("--start-state", type=float, nargs='+', default=[0., 0.],
                        help="Comma-separated list of 3 elements.")  # goal position for PE

    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--eval-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--checkpoint-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="the episodes of evaluation")
    parser.add_argument(
        "--evaluate-checkpoint", action="store_true", default=False, help="to evaluate a particular model"
    )
    parser.add_argument(
        "--checkpoint-number", type=int, default=100, help="to evaluate a checkpoint"
    )


    # First parse the config file argument
    config_args, remaining_argv = parser.parse_known_args()

    # Load the configuration file if specified
    if config_args.config:
        config = load_config(config_args.config)
        arg_defaults = vars(config_args)
        arg_defaults.update(config)
        parser.set_defaults(**arg_defaults)

    # Parse remaining command line arguments
    args = parser.parse_args(remaining_argv)
    # parsing start state
    # args.start_state = args.start_state.split(",")
    print("start state is:", args.start_state)

    # print("args:", args)
    return args


class MonteCarloValueEstimator:
    def __init__(self, args, env, env_id, seed, agent, env_args, run_save_loc, save_val_folder="true_val",
                 gamma=0.99, num_episodes=10):
        self.args = args
        self.env = env
        self.env_id = env_id
        self.seed = seed
        self.agent = agent
        self.gamma = gamma
        self.env_args = env_args
        self.run_save_loc = run_save_loc
        self.save_val_folder = save_val_folder
        self.num_episodes = num_episodes # this is how many episode to roll out for each state
        if not os.path.exists(self.run_save_loc):
            os.makedirs(self.run_save_loc)


    def update_value_function(self, state, curr_val, n_eps):
        save_name = "state_" + "_".join(map(str, state))
        value_function_file_path = os.path.join(self.run_save_loc, self.save_val_folder)
        if not os.path.exists(value_function_file_path):
            os.makedirs(value_function_file_path)
        value_function_file = os.path.join(value_function_file_path, f"{save_name}.json")
        state_to_val = {"state": tuple(state), "val": curr_val, "n_eps": n_eps}

        # Convert tuple keys to string
        str_keys_dict = {str(k): v for k, v in state_to_val.items()}

        # print(f"saving dict of state to val: {str_keys_dict}")

        with open(value_function_file, 'w') as f:
            json.dump(str_keys_dict, f)

    def simulate_episode(self, start_state, agent, env_reset_option, max_steps=500):
        # eval_env = make_env(self.env_id, self.seed, False, run_name=None, env_args=self.env_args)
        obs, _ = self.env.reset(options=env_reset_option)
        done = False
        eps_reward = 0.
        eps_gamma = 1.
        for curr_step in range(max_steps):
            actions = agent.select_action(obs, epsilon=0, greedy=False)
            next_obs, rewards, terminations, truncations, _ = self.env.step(actions[0])
            done = terminations or truncations
            obs = next_obs
            eps_reward += eps_gamma * np.mean(rewards)
            eps_gamma *=self.gamma
            if done:
                break
        # print("episode done!")
        return start_state, eps_reward

    def create_env_reset_option(self, start_state):
        options ={}
        if self.env_id == "DiscreteActionsPendulumEnv-v0":
            options = {"x": start_state[0], "y": start_state[1], "thetadot": start_state[2]}
        elif self.env_id == "PuddleMultiGoals-v0":
            options = {"x": start_state[0], "y": start_state[1]}
        elif self.env_id == "RoomMultiGoals-v0":
            options = {"x": start_state[0], "y": start_state[1]}
        elif self.env_id =="RoomSingleGoals-v0":
            options = {"x": start_state[0], "y": start_state[1]}
        elif self.env_id == "MultiVariedGoalRoomEnv-v0":
            options = {"x": start_state[0], "y": start_state[1]}
        elif self.env_id == "GVFEnv-v0" or self.env_id == "FourRoomsEnv-v0":
            options = {"x": start_state[0], "y": start_state[1]}
        return options



    def mc_compute_val(self, start_state, store_freq=200):
        val_of_state = 0.0
        for eps in range(1, self.num_episodes+1):
            # print(f" start state {start_state} and eps {eps} and val {val_of_state}")
            env_reset_option = self.create_env_reset_option(start_state)
            _, return_ = self.simulate_episode(start_state, self.agent, env_reset_option)
            val_of_state = (val_of_state *(eps-1) + return_)/eps
            if eps % store_freq ==0:
                self.update_value_function(start_state, val_of_state, eps)

        self.update_value_function(start_state, val_of_state, self.num_episodes)
        return start_state, val_of_state


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
    agent = SimpleTargetPolicy(args, goal_type=args.puddle_goal_type)

    env_args = create_env_args(args, args.env_id)
    env = make_env(args.env_id, args.seed, args.capture_video, run_name=None, env_args=env_args)  # training

    start_state = np.array(args.start_state)

    # MC estimator
    mc = MonteCarloValueEstimator(args, env, args.env_id, args.seed, agent, env_args,
                                  args.run_save_loc, save_val_folder="true_val_random_states",num_episodes=200000)
    mc.mc_compute_val(start_state)
    print("done MC for start state:", start_state)

    # close
    env.close()
    agent.env.close()

