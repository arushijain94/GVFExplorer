import argparse
import envs
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
from src.common import load_config, linear_schedule, make_env
from src.multi_dim_dqn import DQNAgent, parse_args
import warnings
import json
import subprocess
import shlex
from src.common import load_config, make_env, create_env_args
import io
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module='gym')

def get_trajectories_agent(agent, env, writer):
    # given a policy, computes the most commonly occuring states acc. to policy distribution
    start_locations = [[0.1, 0.1], [1., 0.1], [0.5, 0.5], [0., 0.9], [0.1, 0.3], [0.3, 0.4], [0.65, 0.1], [0.2, 0.99]]
    for traj in range(len(start_locations)):
        trajectory = []
        obs, _ = env.reset(options= {"x":start_locations[traj][0], "y":start_locations[traj][1]})
        done = False
        max_steps = 200
        curr_step = 0
        while not done and curr_step<=max_steps:
            trajectory.append(obs)
            actions = agent.select_action(obs, epsilon=0, greedy=False)
            next_obs, rewards, terminations, truncations, _ = env.step(actions[0])
            done = terminations or truncations
            obs = next_obs
            curr_step+=1
        trajectory = np.vstack(trajectory)
        plt, fig = env.plot_trajectory(trajectory)
        # Instead of directly logging the figure to WandB, save it to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Load the image from the buffer into PIL
        image = Image.open(buf)

        # Convert the image to RGB (if it's not already in RGB)
        image = image.convert('RGB')

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert the numpy array to PyTorch tensor
        # Permute the dimensions to [C, H, W] as expected by TensorBoard
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        # Use the SummaryWriter to add the figure
        writer.add_image('traj/Trajectory', image_tensor, traj)
        buf.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    agent = DQNAgent(args)
    print("loading agent")
    # load the agent from particular checkpoint
    agent.load_checkpoint(args.checkpoint_number)
    print("the agent is loaded form the checkpoint")

    # writer = SummaryWriter(f"{args.run_save_loc}/trajectories_chk{args.checkpoint_number}")
    env_args = create_env_args(args, args.env_id)
    env = make_env(args.env_id, args.seed, args.capture_video, run_name=None, env_args=env_args)  # training

    # set agent into eval mode
    agent.q_network.eval()
    agent.target_network.eval()

    # plot trajectories
    get_trajectories_agent(agent, env, agent.writer)
    agent.env.close()
    agent.eval_env.close()
    agent.writer.close()
    env.close()
