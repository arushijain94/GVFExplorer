import sys
import argparse

import envs

import random
import gym
import numpy as np
import wandb
import os
from envs.tabularMDP import TabularMDP
import csv
from src.common import namespace_to_dict
from scipy.special import softmax
import src.common as common

def td_control(mdp, env, max_steps, num_episodes, S, A, P, R, init_policy, gamma, alpha, logdir):
    # Q learning control
    pol = init_policy
    R = env.R
    for _ in range(num_episodes):
        q = mdp.get_q(R, gamma, pol)
        perf = np.round(np.mean(np.einsum('sa,sa->s', pol, q)),1)
        np.save(os.path.join(logdir, 'pol_' + str(perf) +'.npy'), np.round(pol,2))
        q += np.random.normal(loc=2, scale=5, size=(S, A))
        q +=0.1
        pol = softmax(q)
        pol = pol/np.linalg.norm(pol, ord=1, keepdims=True, axis=1)
        print("perf:", perf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=10, type=int, help="Number of episodes")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--timeout", default=300, type=int, help="Max episode length of the environment.")
    parser.add_argument("--size", default=10, type=int, help="Side dimension of the grid.")
    parser.add_argument("--logdir", default="logs/grid/try_2", type=str, help="Directory for logging")
    parser.add_argument("--stochastic_env", default=False, type=bool, help="Type of transition F: det; T:stochastic")
    parser.add_argument("--alpha", default=0.05, type=float, help="Learning Rate")
    parser.add_argument("--env_name", default="simple_grid", type=str, help="{simple_grid, tmaze}")

    args = parser.parse_args()
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # common approach to call for envs
    mdp = common.create_env(args.env_name, size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    env = mdp.env

    num_episodes = args.num_episodes
    alpha = args.alpha
    gamma = args.gamma
    init_policy = np.random.uniform(size=(num_states, num_actions))
    init_policy = softmax(init_policy, axis=1)
    print("init policy:", np.round(init_policy,2))

    td = td_control(
        mdp=mdp,
        env=env,
        max_steps=args.timeout,
        num_episodes=args.num_episodes,
        S=num_states,
        A=num_actions,
        P=mdp.P,
        R=mdp.R,
        init_policy=init_policy,
        gamma=gamma,
        alpha=alpha,
        logdir=args.logdir)
