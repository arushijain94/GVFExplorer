### Q:
# 1. How to handle zeros in value functions
# 2. Should we log environment steps?
# 3. Should we also log the density of state visitation distribution?
import os
import sys
import argparse
from src.common import namespace_to_dict, get_csv_logger, get_target_policy, add_value_function
import random
import gym
import numpy as np
from envs.tabularMDP import TabularMDP
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb


def choose_action(state, epsilon, behavior_policy):
    if epsilon > 0 and random.uniform(0,1) <= epsilon:
        return int(random.choice(np.arange(num_actions)))
    else:
        return int(random.choices(np.arange(num_actions), weights=behavior_policy[state])[0])


def update_behavior_policy(env, max_steps, epsilon, theta, num_states, num_actions,
                           num_episodes, gamma, alpha, rho, state_return):
    num_steps = 0
    policy = theta / np.linalg.norm(theta, ord=1, keepdims=True, axis=1)
    update_val = np.zeros_like(theta)
    for _ in range(num_episodes):
        obs = env.reset()
        state = np.argmax(obs)
        done = False
        g = 0
        rho_ = 1.
        gamma_ = 1.0
        env_steps = 0
        vis_states = []
        rewards = []
        gammas = []
        rhos = [rho_]
        visited = np.zeros((num_states, num_actions))
        while not done and env_steps < max_steps:
            env_steps += 1
            vis_states.append(state)
            action = choose_action(state, epsilon, policy)
            obs, r, done, info = env.step(action)
            next_state = np.argmax(obs)
            rho_ *= rho[state, action]
            g += gamma_ * r

            gammas.append(gamma_)
            rewards.append(r)
            rhos.append(rho_)

            gamma_ *= gamma
            visited[state, action] += 1
            state = next_state

        num_steps +=env_steps
        rewards = np.array(rewards)
        gammas = np.array(gammas)
        rhos = np.array(rhos)

        # get corrected return under behavior policy
        for i in range(len(vis_states)):
            state_return[vis_states[i]].append(
                np.sum(rewards[i:] * (gammas[i:] / gammas[i]) * (rhos[i + 1:] / rhos[i])))

        update_val += np.square(g*rho_) * visited * (1. - policy)

    update_val /= num_episodes
    # update behavior policy
    theta = theta + alpha * update_val
    theta += 1e-6

    return theta, num_steps, state_return

#
# def estimate_value_function(env, epsilon, behavior_policy, num_states, num_actions, num_episodes, gamma, alpha, rho,
#                             value_function=None):
#     if value_function is None:
#         value_function = np.zeros((num_states))
#
#     num_steps = 0
#     for _ in range(num_episodes):
#         obs, _ = env.reset()
#         state = np.argmax(obs)
#         done = False
#         while not done:
#             num_steps += 1
#             action = choose_action(state, epsilon, behavior_policy)
#             # action = random.choices(np.arange(num_actions), weights=behavior_policy[state])[0]
#             # action = np.random.choice(np.arange(num_actions), p=behavior_policy[state])
#             obs, r, d1, d2, info = env.step(action)
#             next_state = np.argmax(obs)
#
#             target = r + gamma * rho[state, action] * value_function[next_state]
#             value_function[state] += alpha * (target - value_function[state])
#
#             state = next_state
#             done = d1 or d2
#     return value_function + 1e-6, num_steps


def get_var_s(state, behavior_policy, var_function, rho):
    # var(s) = \sum_a \mu(s,a) \rho^2(s,a) var(s,a)
    return np.sum(behavior_policy[state] * (rho[state, :] ** 2) * var_function[state, :])

#
# def estimate_variance_function(env, epsilon, behavior_policy, value_function, num_states, num_actions, num_episodes,
#                                gamma, alpha, rho, var_function):
#     if var_function is None:
#         var_function = np.zeros((num_states, num_actions))
#
#     for _ in range(num_episodes):
#         obs, _ = env.reset()
#         state = np.argmax(obs)
#         done = False
#         while not done:
#             action = choose_action(state, epsilon, behavior_policy)
#             # action = random.choices(np.arange(num_actions), weights=behavior_policy[state])[0]
#             obs, r, d1, d2, info = env.step(action)
#             next_state = np.argmax(obs)
#             # next_action = np.random.choice(np.arange(num_actions), p=behavior_policy[next_state])
#
#             delta = r + gamma * value_function[next_state] - value_function[state]
#             # expected sarsa
#             target = np.square(delta) + np.square(gamma) * get_var_s(next_state, behavior_policy, var_function, rho)
#             # target = np.square(delta) + np.square(gamma) * np.square(rho[next_state,next_action]) * var_function[next_state, next_action]
#
#             var_function[state, action] += alpha * (target - var_function[state, action])
#
#             state = next_state
#             # action = next_action
#             done = d1 or d2
#
#     return var_function
#

def get_policy_distribution(env, num_states, num_episodes, policy):
    vis_dist = np.zeros(num_states)

    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = np.argmax(obs)
        vis_dist[state] += 1
        done = False
        action = random.choices(np.arange(num_actions), weights=policy[state])[0]
        # action = np.random.choice(np.arange(num_actions), p=policy[state])
        while not done:
            obs, r, d1, d2, info = env.step(action)
            # print (np.argmax(obs), d1, d2)
            state = np.argmax(obs)
            action = random.choices(np.arange(num_actions), weights=policy[state])[0]
            vis_dist[state] += 1

            done = d1 or d2
        # sys.exit(-1)
    return vis_dist / np.sum(vis_dist)


def plot_grid_axis(ax, mat, height, width):
    ax.axis('off')
    ax.pcolormesh(np.flip(mat, axis=0), cmap='Greys')
    for s in range(width + 1):
        ax.axvline(x=s, color='black')
    for s in range(height + 1):
        ax.axhline(y=s, color='black')

    max_val = np.max(mat)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_val)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    mat = np.flip(np.reshape(mat, (height, width)), axis=0)
    pcm = plt.pcolormesh(mat, norm=norm)

    # print (vis.shape)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            ax.text(x + 0.5, y + 0.5, "%.2f" % mat[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='red'
                    )

    # ax.colorbar(pcm, extend='both')


def plot_traj_grids(value_functions, var_functions, behavior_policies, bp_dists, tgt_dists, height, width):
    plt.clf()
    plt.cla()

    scale = 0.2
    color = [1, 0, 0]
    num_plots = len(value_functions)
    plot_width = 11

    fig, axs = plt.subplots(num_plots, plot_width, figsize=(plot_width * (width + 1), num_plots * (height + 1)))
    for i, (value, var, policy, bp_dist, tgt_dist) in enumerate(
            zip(value_functions, var_functions, behavior_policies, bp_dists, tgt_dists)):
        # Plot the Trace
        value = np.reshape(value, (height, width))
        plot_grid_axis(axs[i, 0], value, height, width)

        for a in range(num_actions):
            var_ = np.reshape(var[:, a], (height, width))
            plot_grid_axis(axs[i, a + 1], var_, height, width)

        for a in range(num_actions):
            pol = np.reshape(policy[:, a], (height, width))
            plot_grid_axis(axs[i, a + 5], pol, height, width)

        bp_dist_ = np.reshape(bp_dist, (height, width))
        plot_grid_axis(axs[i, 9], bp_dist_, height, width)

        tgt_dist_ = np.reshape(tgt_dist, (height, width))
        plot_grid_axis(axs[i, 10], tgt_dist_, height, width)

    axs[0, 0].set_title('Value Function', fontsize=30)
    axs[0, 1].set_title('Variance (L)', fontsize=30)
    axs[0, 2].set_title('Variance (D)', fontsize=30)
    axs[0, 3].set_title('Variance (R)', fontsize=30)
    axs[0, 4].set_title('Variance (U)', fontsize=30)
    axs[0, 5].set_title('Policy (L)', fontsize=30)
    axs[0, 6].set_title('policy (D)', fontsize=30)
    axs[0, 7].set_title('Policy (R)', fontsize=30)
    axs[0, 8].set_title('Policy (U)', fontsize=30)
    axs[0, 9].set_title('Behavior Policy Dist', fontsize=30)
    axs[0, 10].set_title('Target Policy Dist', fontsize=30)

    plt.savefig('test_heatmap.png')

    return plt


if __name__ == '__main__':
    '''
    TODO:
    1. Add min clip for action from a policy
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes for tuning PI")
    parser.add_argument("--num_iter", default=10, type=int,
                        help="Total number of iterations of updating the behaviors policy.")
    parser.add_argument("--alpha", default=0.01, type=float, help="Learning Rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="bpg", type=str, help="Group name for Wandb")
    parser.add_argument("--wandb_name", default="test", type=str, help="Log name for Wandb")
    parser.add_argument("--timeout", default=300, type=int, help="Max episode length of the environment.")
    parser.add_argument("--K", default=3, type=int, help="Number of trajectories.")
    parser.add_argument("--rho_max", default=1, type=float, help="Maximum value of Rho allowed.")
    parser.add_argument("--rho_min", default=1e-2, type=float, help="Minimum value of Rho allowed")
    parser.add_argument("--epsilon", default=0.1, type=float, help="epsilon exploration")
    parser.add_argument("--logdir", default="logs/BPG", type=str, help="Directory for logging")
    parser.add_argument("--size", default=5, type=int, help="size of MDP")
    parser.add_argument("--stochastic_env", default=False, type=bool, help="Type of transition F: det; T:stochastic")
    args = parser.parse_args()

    # initializing wandb for storing results
    wandb.init(project="BPG", entity="jainarus", group=args.wandb_group,
               name=args.wandb_name, config=namespace_to_dict(args),
               mode="online")
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # deterministic 5X5 MDP
    mdp = TabularMDP(size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    env = mdp.env

    num_iter = args.num_iter
    num_episodes = args.num_episodes
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon

    target_policy = get_target_policy(num_states)
    # true value function of target_pi
    vf = mdp.get_v(target_policy)
    target_sample_means = np.array(vf)

    # target policy state distribution
    mu_s = mdp.get_state_distribution_of_policy(target_policy)
    mu_s = mu_s / np.sum(mu_s)

    #### Behavior
    behavior_policy = np.copy(target_policy)
    theta = np.copy(target_policy)
    # value_functions, var_functions, behavior_policies, bp_dists, tgt_dists = [], [], [], [], []

    value_function = None
    var_function = None
    rho = target_policy / behavior_policy
    rho = np.clip(rho, args.rho_min, args.rho_max)

    # csv
    header = ["num_samples", "Target(mean)", "Var(mean)", "Bias(mean)", "MSE(mean)",
              "Target(target_dist)", "Var(target_dist)", "Bias(target_dist)", "MSE(target_dist)",
              "Target(init_dist)", "Var(init_dist)", "Bias(init_dist)", "MSE(init_dist)"]

    csv_file, csv_writer = get_csv_logger(args.logdir)
    csv_writer.writerow(header)  # write header
    csv_writer.writerow([0.0] * len(header))

    state_return = [[0.0] for _ in range(num_states)]
    total_env_steps = 0

    # maintains the list V(s) of all the value functions seen so far
    state_values = [[0.0] for _ in range(num_states)]

    for iter_ in range(num_iter):
        metrics = {}
        state_return = [[0.0] for _ in range(num_states)]
        # Var and Value update for Behavior policy
        theta, num_steps, state_return = update_behavior_policy(env, args.timeout, epsilon, theta, num_states, num_actions, args.K,
                                                  gamma, alpha, rho, state_return)
        behavior_policy = theta / np.linalg.norm(theta, ord=1, keepdims=True, axis=1)
        total_env_steps += num_steps

        state_values = add_value_function(state_values, state_return)
        # print("state values at iter:", iter_, "is: ", state_values)

        rho = target_policy / behavior_policy
        rho = np.clip(rho, args.rho_min, args.rho_max)

        if iter_ %2 ==0 and iter_!=0:
            variances = [np.var(x) for x in state_values]
            # print("var:", variances)
            behavior_sample_means = [np.mean(x) for x in state_values]
            biases = np.abs(target_sample_means - np.array(behavior_sample_means))
            MSE = [np.mean(np.square(state_values[s] - target_sample_means[s])) for s in range(num_states)]

            metrics["num_samples"] = total_env_steps
            metrics["Var(mean)"] = np.mean(np.array(variances))
            metrics["Bias(mean)"] = np.mean(np.array(biases))
            metrics["MSE(mean)"] = np.mean(np.array(MSE))
            metrics["Target(mean)"] = np.mean(target_sample_means)

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
                        metrics["Target(mean)"], metrics["Var(mean)"], metrics["Bias(mean)"], metrics["MSE(mean)"],
                        metrics["Target(target_dist)"], metrics["Var(target_dist)"], metrics["Bias(target_dist)"],
                        metrics["MSE(target_dist)"],
                        metrics["Target(init_dist)"], metrics["Var(init_dist)"], metrics["Bias(target_dist)"],
                        metrics["MSE(init_dist)"]]
            csv_writer.writerow(data_row)

            # wandb logging
            wandb.log(metrics)
            csv_file.flush()


    # log beh policy
    if iter_ % 1000 == 0:
        np.save(os.path.join(args.logdir, "beh_" + str(total_env_steps) + ".npy"), behavior_policy)


    wandb.finish()
    csv_file.close()
    np.save(os.path.join(args.logdir, "final_beh_" + str(total_env_steps) + ".npy"), behavior_policy)

