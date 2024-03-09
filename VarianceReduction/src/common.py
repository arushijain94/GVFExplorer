import argparse
import numpy as np
import os
import csv
from envs.tabularMDP import TabularMDP
from envs.tmaze import TMaze
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LinearScheduleClass(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class Actions:
    UP = 2
    RIGHT = 1
    DOWN = 3
    LEFT = 0

def set_seed(seed):
    # Set the random seed to a specific value, e.g., 42
    np.random.seed(seed)

def get_target_policy(num_states):
    target_policy = np.tile([0.325, 0.175, 0.325, 0.175], (num_states, 1))
    return target_policy

def get_mult_simple_target_policy(num_states, num_actions):
    target_policy1 = np.tile([0.45, 0.05, 0.05, 0.45], (num_states, 1))
    target_policy2 = np.tile([0.3, 0.15, 0.3, 0.25], (num_states, 1))
    target_policy = np.array([target_policy1, target_policy2])
    return target_policy

def create_multiple_reward(n_target_policies, size, S, A, env_name, cumulant_goal3=1):
    # time_step=0 -> initial calling
    # else time_step would operate acc. to internal time steps
    if env_name == "simple_grid":
        #same reward function
        R = np.zeros((n_target_policies, S, A))
        R[:, size*size-1, :] = +100
        cumulant_goal3=0
        # R = np.zeros((n_target_policies, S, A))
        # R[0, size * size - 1, :] = +10
        # new_reward = get_drifter_reward(cumulant_goal3)
        # cumulant_goal3 = new_reward
        # R[1, size * size - 1 - (size-1), :] = new_reward
    if env_name == "gvfs_grid":
        #same reward function
        R = np.zeros((n_target_policies, S, A))
        R[0, size*size -size, :] = +100 # bottom left
        R[1, size * size - 1, :] = +100 # bottom right
        cumulant_goal3=0
    if env_name == "tmaze":
        # different reward functions
        R = np.zeros((n_target_policies, S, A))
        for i in range(1, 1+n_target_policies):
            R[i - 1, :, :],  info = get_tmaze_reward(S, A, cumulant_goal3, goal_number=i)
            if i==3:
                cumulant_goal3 = info
    return R, cumulant_goal3

# def plot_mat_single_goal(mat, size, num_steps, title):
#     fig, ax = plt.subplots()
#     heatmap = ax.imshow(mat, origin='upper', cmap='hot', interpolation='nearest')
#     # Adding grid lines
#     ax.set_xticks(np.arange(-.5, size, 1), minor=True)
#     ax.set_yticks(np.arange(-.5, size, 1), minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
#     ax.tick_params(which="minor", size=0)
#     # add the goals
#     circle = patches.Circle((0,0), radius=0.4, edgecolor='green',
#                             facecolor='green',
#                             label='Goals')
#     ax.add_patch(circle)
#     ax.set_aspect('equal', adjustable='box')
#     fig.colorbar(heatmap, ax=ax, label=title)
#     ax.set_title(f'Steps:{num_steps}')
#     return plt

def plot_mat_single_dual_goal(env_name, mat, size, num_steps, title, vmin=4, vmax=4, show_v_limit=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    if show_v_limit:
        heatmap = ax.imshow(mat, origin='upper', cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    else:
        heatmap = ax.imshow(mat, origin='upper', cmap='hot', interpolation='nearest')
    # Adding grid lines
    # Step 3: Adjust the ticks and grid
    # Set the ticks to be at every integer value
    ax.set_xticks(np.arange(0, size, 1))
    ax.set_yticks(np.arange(0, size, 1))

    ax.set_xticks(np.arange(-.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, size, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    # add the goals
    if env_name == "online_rooms_different_reward_grid" or env_name=="semigreedy_online_rooms_different_reward_grid":
        circle = patches.Circle((0,0), radius=0.4, edgecolor='green',
                                facecolor='green',
                                label='Goals')
        ax.add_patch(circle)
        circle2 = patches.Circle((size-1,0), radius=0.4, edgecolor='green',
                                facecolor='green',
                                label='Goals')
        ax.add_patch(circle2)
    if env_name == "online_rooms_same_reward_grid" or env_name== "semigreedy_online_rooms_same_reward_grid":
        circle = patches.Circle((0, 0), radius=0.4, edgecolor='green',
                                facecolor='green',
                                label='Goals')
        ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.colorbar(heatmap, ax=ax, shrink=0.7)
    plt.tight_layout()
    # ax.set_title(f'Steps:{num_steps}')
    return plt


# def plot_mat_dual_goal(mat, size, num_steps, title):
#     fig, ax = plt.subplots()
#     heatmap = ax.imshow(mat, origin='upper', cmap='hot', interpolation='nearest')
#     # Adding grid lines
#     ax.set_xticks(np.arange(-.5, size, 1), minor=True)
#     ax.set_yticks(np.arange(-.5, size, 1), minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
#     ax.tick_params(which="minor", size=0)
#     # add the goals
#     circle = patches.Circle((0,0), radius=0.4, edgecolor='green',
#                             facecolor='green',
#                             label='Goals')
#     ax.add_patch(circle)
#     circle2 = patches.Circle((0, size-1), radius=0.4, edgecolor='green',
#                             facecolor='green',
#                             label='Goals')
#     ax.add_patch(circle2)
#     ax.set_aspect('equal', adjustable='box')
#     fig.colorbar(heatmap, ax=ax, label=title)
#     ax.set_title(f'Steps:{num_steps}')
#     return plt

def get_tmaze_reward(S, A, cumulant_goal3, goal_number):
    R = np.zeros((S, A))
    if goal_number ==2: # constant reward
        R[28,:] = 1
    elif goal_number ==4:# constant reward
        R[34,:] = 1
    elif goal_number ==1:# distractor reward C_i^t = N(mu_i, aigma_i); mu_i=1, sigma^2 =25
        R[0,:] = np.random.normal(loc=1, scale=5., size=1)[0]
    elif goal_number ==3:# driftwer reward C_i^t = C_i^{t-1} + N(mu_i, aigma_i); mu_i=1, sigma^2 =0.01
        new_reward = get_drifter_reward(cumulant_goal3)
        R[6,:] = new_reward
        cumulant_goal3 = new_reward
        # print("g3:", new_reward)
    else:
        print("Wrong goal number in Tmaze - check common.py!!")
    return (R, cumulant_goal3)

def get_drifter_reward(cumulant_goal3):
    return cumulant_goal3 + np.random.normal(loc=0, scale=0.1, size=1)[0]

def get_scaled_up_target_policies(num_states, env_name,num_target_pols=8):
    target_policies = np.ones((num_target_pols, num_states, 4))
    if env_name == "grid_scaled_up_exp":
        target_policies[0, :] = np.tile(np.array([0.1, 0.1, 0.7, 0.1]), (num_states, 1))  # N
        target_policies[1, :] = np.tile(np.array([0.1, 0.7, 0.1, 0.1]), (num_states, 1))  # E
        target_policies[2, :] = np.tile(np.array([0.1, 0.1, 0.1, 0.7]), (num_states, 1))  # S
        target_policies[3, :] = np.tile(np.array([0.7, 0.1, 0.1, 0.1]), (num_states, 1))  # W
    # directions L, R, U , D
    # if env_name == "grid_scaled_up_exp":
    #     target_policies[0, :] = np.tile(np.array([0.1, 0.1, 0.7, 0.1]), (num_states, 1)) # N
    #     target_policies[1, :] = np.tile(np.array([0.1, 0.4, 0.4, 0.1]), (num_states, 1))  # NE
    #     target_policies[2, :] = np.tile(np.array([0.1, 0.7, 0.1, 0.1]), (num_states, 1))  # E
    #     target_policies[3, :] = np.tile(np.array([0.1, 0.4, 0.1, 0.4]), (num_states, 1))  # SE
    #     target_policies[4, :] = np.tile(np.array([0.1, 0.1, 0.1, 0.7]), (num_states, 1))  # S
    #     target_policies[5, :] = np.tile(np.array([0.4, 0.1, 0.1, 0.4]), (num_states, 1))  # SW
    #     target_policies[6, :] = np.tile(np.array([0.7, 0.1, 0.1, 0.1]), (num_states, 1))  # W
    #     target_policies[7, :] = np.tile(np.array([0.4, 0.1, 0.4, 0.1]), (num_states, 1))  # NW
    #     #
    #     # target_policies[0, :] = np.tile(np.array([0.175, 0.175, 0.25, 0.4]), (num_states, 1))
    #     # target_policies[1, :] = np.tile(np.array([0.25, 0.15, 0.25, 0.35]), (num_states, 1))
    #     # target_policies[2, :] = np.tile(np.array([0.35, 0.15, 0.35, 0.15]), (num_states, 1))
    #     # target_policies[3, :] = np.tile(np.array([0.35, 0.15, 0.15, 0.35]), (num_states, 1))
    #     # target_policies[4, :] = np.tile(np.array([0.15, 0.35, 0.15, 0.35]), (num_states, 1))
    #     # target_policies[5, :] = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (num_states, 1))
    #     # target_policies[6, :] = np.tile(np.array([0.2, 0.2, 0.4, 0.2]), (num_states, 1))
    #     # target_policies[7, :] = np.tile(np.array([0.45, 0.45, 0.05, 0.05]), (num_states, 1))
    #     # target_policies[8, :] = np.tile(np.array([0.05, 0.05, 0.45, 0.45]), (num_states, 1))
    #     # target_policies[9, :] = np.tile(np.array([0.25, 0.2, 0.4, 0.15]), (num_states, 1))
    return target_policies


def get_target_policies(num_states, env_name, num_target_pols=3):
    target_policies = np.ones((num_target_pols, num_states, 4))
    if env_name == "simple_grid":
        target_policies[0,:] = np.tile(np.array([0.325, 0.175, 0.325, 0.175]), (num_states, 1))
        target_policies[1, :] = np.tile(np.array([0.5, 0.2, 0.2, 0.1]), (num_states, 1))
        target_policies[2, :] = np.tile(np.array([0.3, 0.175, 0.4, 0.125]), (num_states, 1))
    if env_name == "gvfs_grid":
        target_policies[0, :] = np.tile(np.array([0.2, 0.15, 0.5, 0.15]), (num_states, 1))
        target_policies[1, :] = np.tile(np.array([0.15, 0.2, 0.5, 0.15]), (num_states, 1))
        # target_policies[2, :] = np.tile(np.array([0.25, 0.35, 0.25, 0.15]), (num_states, 1))
    if env_name == "online_rooms_same_reward_grid" or env_name == "online_rooms_different_reward_grid":
        target_policies[0, :] = np.tile(np.array([0.175, 0.175, 0.25, 0.4]), (num_states, 1))
        target_policies[1, :] = np.tile(np.array([0.25, 0.15, 0.25, 0.35]), (num_states, 1))
    if env_name == "semigreedy_online_rooms_same_reward_grid" or env_name == "semigreedy_online_rooms_different_reward_grid":
        # target_policies[0, :] = np.tile(np.array([0.25, 0.2, 0.4, 0.15]), (num_states, 1))
        # target_policies[1, :] = np.tile(np.array([0.15, 0.25, 0.35, 0.25]), (num_states, 1))
        target_policies[0, :] = np.tile(np.array([0.4, 0.1, 0.4, 0.1]), (num_states, 1))
        target_policies[1, :] = np.tile(np.array([0.1, 0.4, 0.4, 0.1]), (num_states, 1))
    return target_policies


def create_env(name, size=5, gamma=0.99, stochastic_transition=0):
    size= int(size)
    print("name of env:", name)
    if name == "online_rooms_same_reward_grid" or name=="semigreedy_online_rooms_same_reward_grid":
        terminal_states = [(0,0)]
        return TabularMDP(terminal_states, size=size, gamma=gamma, env_name=name,
                          stochastic_transition=stochastic_transition)
    if name == "online_rooms_different_reward_grid" or name=="semigreedy_online_rooms_different_reward_grid":
        terminal_states = [(0, 0), (0, size-1)]
        return TabularMDP(terminal_states, size=size, gamma=gamma, env_name=name,
                          stochastic_transition=stochastic_transition)
    if name == "simple_grid":
        terminal_states = [(size - 1, size - 1)]
        print("env: Tabular Grid")
        return TabularMDP(terminal_states, size=size, gamma=gamma, env_name=name, stochastic_transition=stochastic_transition)
    if name =="gvfs_grid":
        terminal_states = [(size-1, 0), (size-1, size-1)]
        return TabularMDP(terminal_states, size=size, gamma=gamma, env_name=name, stochastic_transition=stochastic_transition)
    if name == "tmaze":
        print("env: TMaze")
        return TMaze() # no terminal state
    if name =="grid_scaled_up_exp":
        terminal_states = get_terminal_state_for_scaled_exp(size)
        return TabularMDP(terminal_states, size=size, gamma=gamma, env_name=name,
                          stochastic_transition=stochastic_transition)
    else:
        print("Env name does not exists -- file common.py!!!")
        sys.exit(-1)

def get_terminal_state_for_scaled_exp(size):
    r_dict = get_mean_reward_scaled_exp()
    r_states = list(r_dict.keys())
    terminal_states_list =[]
    for s in r_states:
        x,y = s//size,s%size
        terminal_states_list.append((x,y))
    print("ter state:", terminal_states_list)
    return terminal_states_list


def get_mean_reward_scaled_exp(size=20):
    # writing R as dict = {'state ind': reward value}
    # positions = [(4,4), (3,18),(10,18),(18,7),(19,19)]
    # R = {
    #     23: 55,
    #     50 : 62,
    #     84: 50,
    #     78: 80,
    #     105:100,
    #     120: 100,
    #     179: 72,
    #     200: 57,
    #     203: 92,
    #     212: 95,
    #     256: 83,
    #     288: 59,
    #     310: 85,
    #     367: 98,
    #     388: 66,
    #     399: 99,
    # }
    R = {
        1: 50,
        24: 69,
        45: 70,
        130: 80,
        206: 65,
        287: 31,
        303: 78,
        378: 85,
        384: 92,
        399: 100,
    }

    # R = {
    #     1: 50,
    #     4: 69,
    #     25: 70,
    #     30: 80,
    #     46: 65,
    #     55: 31,
    #     73: 78,
    #     78: 85,
    #     84: 92,
    #     91: 84,
    # }
    # R={1:100}

    return R




def get_mean_reward(env_name, n_target_policies, S, A, size):
    R = np.zeros((n_target_policies, S, A))
    if env_name == "online_rooms_same_reward_grid" or env_name == "semigreedy_online_rooms_same_reward_grid":
        R[:, 0, :] = 100
        return R
    if env_name == "online_rooms_different_reward_grid" or env_name == "semigreedy_online_rooms_different_reward_grid":
        R[0, 0, :] = 100
        R[1, size-1, :] = 50
        return R
    return R

def get_reward_at_each_step(env_name, n_target_policies, S, A, size, s_t, a_t, std_dev_reward = 5):
    R = get_mean_reward(env_name, n_target_policies, S, A, size)
    curr_r = []
    for i in range(n_target_policies):
        r_s_a = R[i,s_t,a_t]
        if r_s_a == 0:
            curr_r.append(0)
        else:
            reward = np.random.normal(loc=r_s_a, scale=std_dev_reward)
            curr_r.append(reward)
    return np.array(curr_r)


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def get_csv_logger(dir):
    csv_file = open(os.path.join(dir, "log.txt"), mode='w')
    csv_writer = csv.writer(csv_file)
    return csv_file, csv_writer

def add_value_function(state_values, state_returns):
    for i in range(len(state_values)):
        state_values[i].append(np.mean(state_returns[i]))  # adding avg. return to value function list
    return state_values

def smart_clipping(x, x_min, x_max):
    x[x <= 0] = 0.
    x[(x > 0) & (x <x_min)] = x_min
    x[(x > 1)] = x_max
    return x

def set_rho(target_policy, behavior_policy, rho_min, rho_max):
    zero_denominator_mask = (behavior_policy ==0)
    rho = np.zeros_like(target_policy)
    rho[~zero_denominator_mask] = target_policy[~zero_denominator_mask] / behavior_policy[~zero_denominator_mask]
    rho = smart_clipping(rho, rho_min, rho_max)
    return rho

def evaluate_montecarlo_correct(env, num_actions, max_steps, num_episodes, policy, rho, gamma, cor_return,
                                return_values_beh):
    env_steps = 0

    for _ in range(num_episodes):
        obs = env.reset()
        gamma_ = 1
        rho_ = 1
        ret = 0
        state = np.argmax(obs)
        done = False
        action = int(np.random.choices(np.arange(num_actions), weights=policy[state])[0])
        vis_states = []
        rewards = []
        gammas = []
        rhos = [rho_]
        while not done and env_steps < max_steps:
            env_steps += 1
            vis_states.append(state)
            obs, r, done, info = env.step(action)
            rho_ *= rho[state, action]
            state = np.argmax(obs)
            action = int(np.random.choices(np.arange(num_actions), weights=policy[state])[0])
            gammas.append(gamma_)
            rewards.append(r)
            rhos.append(rho_)
            gamma_ *= gamma

        rewards = np.array(rewards)
        gammas = np.array(gammas)
        rhos = np.array(rhos)
        # Do the n-step thing with gamma + for rho values
        for i in range(len(vis_states)):
            cor_return[vis_states[i]].append(
                np.sum(rewards[i:] * (gammas[i:] / gammas[i]) * (rhos[i + 1:] / rhos[i])))
            return_values_beh[vis_states[i]].append(
                np.sum(rewards[i:] * (gammas[i:] / gammas[i])))

    return cor_return, return_values_beh, env_steps
