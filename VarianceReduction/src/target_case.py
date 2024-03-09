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
from src.common import get_target_policy, namespace_to_dict

def evaluate_montecarlo_correct(env, num_states, max_steps, num_episodes, policy, rho, gamma, state_return):

    steps=0
    for _ in range(num_episodes):
        # obs, _ = env.reset()
        obs = env.reset()
        gamma_ = 1
        rho_ = 1
        state = np.argmax(obs)
        done = False
        action = int(random.choices(np.arange(num_actions), weights=policy[state])[0])

        vis_states = []
        rewards = []
        gammas = []
        rhos = [rho_]
        env_steps = 0
        while not done and env_steps < max_steps:
            env_steps += 1
            vis_states.append(state)
            obs, r, done, info = env.step(action)
            rho_ *= rho[state, action]
            state = np.argmax(obs)
            action = int(random.choices(np.arange(num_actions), weights=policy[state])[0])
            # done = d1 or d2

            gammas.append(gamma_)
            rewards.append(r)
            rhos.append(rho_)

            gamma_ *= gamma
        steps+=env_steps
        rewards = np.array(rewards)
        gammas = np.array(gammas)
        rhos = np.array(rhos)
        # Do the n-step thing with gamma + for rho values
        for i in range(len(vis_states)):
            state_return[vis_states[i]].append(
                np.sum(rewards[i:] * (gammas[i:] / gammas[i]) * (rhos[i + 1:] / rhos[i])))

    return state_return, steps


def get_csv_logger(dir):
    csv_file = open(os.path.join(dir, "log.txt"), mode='w')
    csv_writer = csv.writer(csv_file)
    return csv_file, csv_writer


def get_policy_distribution(env, num_states, num_episodes, policy):
    vis_dist = np.zeros(num_states)

    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = np.argmax(obs)
        vis_dist[state] += 1
        done = False
        action = random.choices(np.arange(num_actions), weights=policy[state])[0]
        while not done:
            obs, r, d1, d2, info = env.step(action)
            # print (np.argmax(obs), d1, d2)
            state = np.argmax(obs)
            action = random.choices(np.arange(num_actions), weights=policy[state])[0]
            vis_dist[state] += 1

            done = d1 or d2
        # sys.exit(-1)
    return vis_dist / np.sum(vis_dist)

def add_value_function(state_values, state_returns):
    for i in range(len(state_values)):
        state_values[i].append(np.mean(state_returns[i]))  # adding avg. return to value function list
    return state_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes for tuning PI")
    parser.add_argument("--num_iter", default=5, type=int,
                        help="Total number of iterations of updating the behaviors policy.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="gridworld", type=str, help="Group name for Wandb")
    parser.add_argument("--wandb_name", default="test", type=str, help="Log name for Wandb")
    parser.add_argument("--timeout", default=300, type=int, help="Max episode length of the environment.")
    parser.add_argument("--size", default=5, type=int, help="Side dimension of the grid.")
    parser.add_argument("--logdir", default="logs/target", type=str, help="Directory for logging")
    parser.add_argument("--stochastic_env", default=False, type=bool, help="Type of transition F: det; T:stochastic")

    args = parser.parse_args()

    wandb.init(project="BPG", entity="jainarus", group=args.wandb_group,
               name=args.wandb_name, config=namespace_to_dict(args),
               mode="online")

    # logging data
    log_dir = args.logdir #os.path.join(args.logdir, args.wandb_group, args.wandb_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # deterministic 5X5 MDP
    mdp = TabularMDP(size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    env = mdp.env

    num_iter = args.num_iter
    num_episodes = args.num_episodes
    gamma = args.gamma

    # target policy
    target_pi = get_target_policy(num_states)
    # print("target pol:", target_pi)


    # true value function of target_pi
    vf = mdp.get_v(target_pi)
    target_sample_means = vf
    # print("True V Pi:", vf)

    # target policy state distribution
    mu_s = mdp.get_state_distribution_of_policy(target_pi)
    mu_s = mu_s / np.sum(mu_s)


    # maintains the list G(s) seen so far after every evaluation episode
    state_return = [[0.0] for _ in range(num_states)]
    total_env_steps = 0

    # maintains the list V(s) of all the value functions seen so far
    state_values = [[0.0] for _ in range(num_states)]

    header = ["num_samples", "Target(mean)","Var(mean)", "Bias(mean)", "MSE(mean)",
              "Target(target_dist)", "Var(target_dist)", "Bias(target_dist)","MSE(target_dist)",
              "Target(init_dist)",  "Var(init_dist)", "Bias(init_dist)", "MSE(init_dist)"]

    csv_file, csv_writer = get_csv_logger(log_dir)
    csv_writer.writerow(header)  # write header
    csv_writer.writerow([0.0] * len(header))

    for iter_ in range(num_iter):
        metrics = {}

        # Value Update
        state_return, new_steps = evaluate_montecarlo_correct(env, num_states, args.timeout, num_episodes, target_pi,
                                                              np.ones((num_states, num_actions)), gamma, state_return)
        total_env_steps += new_steps

        state_values = add_value_function(state_values, state_return)
        # print("state values at iter:", iter_, "is: ", state_values)


        if iter_ % 10 ==0 and iter_!=0:
            # compute metrics
            variances = [np.var(x) for x in state_values] # var in value functions
            biases = [np.mean(x) for x in state_values]  # var in value functions
            biases = np.abs(np.array(target_sample_means) - np.array(biases))
            # print("var:", variances)
            # print("bias", biases)
            MSE = [np.mean(np.square(state_values[s] - target_sample_means[s])) for s in range(num_states)]

            target_sample_means = np.array(target_sample_means)

            metrics["num_samples"] = total_env_steps
            # adding var metrics
            # var = \sum_s var(s)/|S|
            metrics["Var(mean)"] = np.mean(np.array(variances))
            metrics["Bias(mean)"] = np.mean(np.array(biases))
            metrics["MSE(mean)"] = np.mean(np.array(MSE))
            metrics["Target(mean)"]= np.mean(target_sample_means)

            # var = \sum_s \mu(s) var(s)
            metrics["Var(target_dist)"] = np.sum(variances * mu_s)
            metrics["Bias(target_dist)"] = np.sum(biases * mu_s)
            metrics["MSE(target_dist)"] = np.sum(MSE * mu_s)
            metrics["Target(target_dist)"] = np.sum(target_sample_means * mu_s)

            # var = \sum_s p0(s) var(s)
            metrics["Var(init_dist)"] = np.sum(variances * mdp.p0)
            metrics["Bias(init_dist)"] = np.sum(biases * mdp.p0)
            metrics["MSE(init_dist)"] = np.sum(MSE * mdp.p0)
            metrics["Target(init_dist)"] = np.sum(target_sample_means * mdp.p0)

            # write row in csv file
            data_row = [metrics["num_samples"],
                        metrics["Target(mean)"],metrics["Var(mean)"], metrics["Bias(mean)"], metrics["MSE(mean)"],
                        metrics["Target(target_dist)"],metrics["Var(target_dist)"], metrics["Bias(target_dist)"], metrics["MSE(target_dist)"],
                        metrics["Target(init_dist)"],metrics["Var(init_dist)"], metrics["Bias(target_dist)"], metrics["MSE(init_dist)"]]
            # print("data row:", data_row)
            csv_writer.writerow(data_row)

            # wandb logging
            wandb.log(metrics)
            if iter_ % 50 == 0:
                csv_file.flush()

    wandb.finish()
    csv_file.close()
