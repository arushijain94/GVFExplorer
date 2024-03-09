### Q:
# 1. How to handle zeros in value functions
# 2. Should we log environment steps?
# 3. Should we also log the density of state visitation distribution?
import os
import sys
import argparse
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from envs.tabularMDP import TabularMDP
import csv
from src.common import namespace_to_dict, get_csv_logger, get_target_policy, add_value_function, set_rho

class TD:
    def __init__(self, env, epsilon, max_steps, num_episodes, S, A, P, R, target_pol, gamma, alpha, rho_min, rho_max, val_func=None,
                 var_func=None):
        self.env = env
        self.epsilon= epsilon
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.epsilon=epsilon
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.target_pol = target_pol
        self.val_func = val_func
        self.var_func = var_func
        if val_func == None:
            self.val_func = np.zeros((self.S, self.A))  # Q(s,a) func
        if var_func == None:
            self.var_func = np.zeros((self.S, self.A)) # var(s,a)
        self.s_a_list = self.get_s_a_pairs()
        self.len_s_a_list = len(self.s_a_list)

    def get_s_a_pairs(self):
        return list(itertools.product(range(self.S), range(self.A)))

    def set_beh_policy(self, policy):
        self.behavior_pol = policy
        self.rho = set_rho(self.target_pol, self.behavior_pol, self.rho_min, self.rho_max)

    def update_val_func(self, s_t, a_t, s_tp1):
        # expected sarsa
        delta = self.R[s_t, a_t] + self.gamma * self.get_val_s(s_tp1) - self.val_func[s_t, a_t]
        self.val_func[s_t, a_t] += self.alpha * delta

    # def update_val_func(self, s_t, a_t):
    #     s_tp1 = np.random.choice(self.S, p=self.P[s_t, a_t])
    #     # expected sarsa
    #     delta = self.R[s_t, a_t] + self.gamma * self.get_val_s(s_tp1) - self.val_func[s_t, a_t]
    #     self.val_func[s_t, a_t] += self.alpha * delta

    def update_val(self, s_a_s_next):
        for i in range(len(s_a_s_next)):
            s, a, s_next = s_a_s_next[i]
            self.update_val_func(s, a, s_next)

    def get_var_s(self, state):
        # var(s) = \sum_a \mu(s,a) \rho^2(s,a) var(s,a)
        return np.sum(self.behavior_pol[state] * (self.rho[state, :] ** 2) * self.var_func[state, :])

    def get_val_s(self, state):
        # v(s) = \sum_a \pi(s,a) q(s,a)
        return np.sum(self.target_pol[state] * self.val_func[state, :])

    def update_var_func(self, s_t, a_t, s_tp1):
        # s_tp1 = np.random.choice(self.S, p=self.P[s_t, a_t])
        delta_v = self.R[s_t, a_t] + self.gamma * self.get_val_s(s_tp1) - self.val_func[s_t, a_t]
        delta_var = (delta_v ** 2.0) + (self.gamma ** 2.0) * self.get_var_s(s_tp1) - self.var_func[s_t, a_t]
        self.var_func[s_t, a_t] += self.alpha * delta_var

    def update_var(self, s_a_s_next):
        for i in range(len(s_a_s_next)):
            s, a, s_next = s_a_s_next[i]
            self.update_var_func(s, a, s_next)

    def update(self, behavior_policy):
        self.set_beh_policy(behavior_policy)
        step = 0
        s_a_s_next_list = []
        for e in range(self.num_episodes):
            obs = self.env.reset()
            s= np.argmax(obs)
            done = False
            env_steps = 0
            while not done and env_steps < self.max_steps:
                if random.random()<= self.epsilon:
                    a = np.random.choice(self.A)
                else:
                    a = np.random.choice(self.A, p=self.behavior_pol[int(s)])
                obs_next, r, done, info = env.step(a)
                s_next = np.argmax(obs_next)
                step +=1
                s_a_s_next = (int(s),int(a),int(s_next))
                s_a_s_next_list.append(s_a_s_next)
                s = s_next
                env_steps+=1
                # print("s{0}, a {1}, s_next {2}, r {3}", s, a, s_next, r)
        self.update_val(s_a_s_next_list)
        random.shuffle(s_a_s_next_list)
        self.update_var(s_a_s_next_list)
        return step

    def get_var(self):
        return self.var_func

    def get_val(self):
        return self.val_func


def choose_action(state, epsilon, behavior_policy):
    if epsilon > 0 and random.uniform(a=0, b=1) < epsilon:
        return random.choice(np.arange(num_actions))
    else:
        return random.choices(np.arange(num_actions), weights=behavior_policy[state])[0]


def estimate_val_var_function(env, epsilon, behavior_policy, num_states, num_actions, num_episodes, gamma, alpha, rho,
                              value_function=None, var_function=None):
    if value_function is None:
        value_function = np.ones((num_states)) * 1e-6

    if var_function is None:
        var_function = np.ones((num_states, num_actions)) * 1e-6

    num_steps = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = np.argmax(obs)
        done = False
        while not done:
            num_steps += 1
            action = choose_action(state, epsilon, behavior_policy)
            obs, r, d1, d2, info = env.step(action)
            next_state = np.argmax(obs)

            # value update
            delta_v = r + gamma * rho[state, action] * value_function[next_state] - value_function[state]
            value_function[state] += alpha * delta_v

            # variance update
            delta_var = np.square(delta_v) + np.square(gamma) * get_var_s(next_state, behavior_policy, var_function,
                                                                          rho) - var_function[state, action]
            var_function[state, action] += alpha * delta_var

            state = next_state
            done = d1 or d2
    return value_function, var_function, num_steps


def estimate_value_function(env, epsilon, behavior_policy, num_states, num_actions, num_episodes, gamma, alpha, rho,
                            value_function=None):
    if value_function is None:
        value_function = np.zeros((num_states))

    num_steps = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = np.argmax(obs)
        done = False
        while not done:
            num_steps += 1
            action = choose_action(state, epsilon, behavior_policy)
            # action = random.choices(np.arange(num_actions), weights=behavior_policy[state])[0]
            # action = np.random.choice(np.arange(num_actions), p=behavior_policy[state])
            obs, r, d1, d2, info = env.step(action)
            next_state = np.argmax(obs)

            target = r + gamma * rho[state, action] * value_function[next_state]
            value_function[state] += alpha * (target - value_function[state])

            state = next_state
            done = d1 or d2
    return value_function + 1e-6, num_steps


def get_var_s(state, behavior_policy, var_function, rho):
    # var(s) = \sum_a \mu(s,a) \rho^2(s,a) var(s,a)
    return np.sum(behavior_policy[state] * (rho[state, :] ** 2) * var_function[state, :])


def estimate_variance_function(env, epsilon, behavior_policy, value_function, num_states, num_actions, num_episodes,
                               gamma, alpha, rho, var_function):
    if var_function is None:
        var_function = np.zeros((num_states, num_actions))

    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = np.argmax(obs)
        done = False
        while not done:
            action = choose_action(state, epsilon, behavior_policy)
            # action = random.choices(np.arange(num_actions), weights=behavior_policy[state])[0]
            obs, r, d1, d2, info = env.step(action)
            next_state = np.argmax(obs)
            # next_action = np.random.choice(np.arange(num_actions), p=behavior_policy[next_state])

            delta = r + gamma * value_function[next_state] - value_function[state]
            # expected sarsa
            target = np.square(delta) + np.square(gamma) * get_var_s(next_state, behavior_policy, var_function, rho)
            # target = np.square(delta) + np.square(gamma) * np.square(rho[next_state,next_action]) * var_function[next_state, next_action]

            var_function[state, action] += alpha * (target - var_function[state, action])

            state = next_state
            # action = next_action
            done = d1 or d2

    return var_function

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
    # print("ax length:", axs.shape)
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
    parser.add_argument("--num_iter", default=5, type=int,
                        help="Total number of iterations of updating the behaviors policy.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="gridworld", type=str, help="Group name for Wandb")
    parser.add_argument("--wandb_name", default="test", type=str, help="Log name for Wandb")
    parser.add_argument("--timeout", default=300, type=int, help="Max episode length of the environment.")
    parser.add_argument("--rho_max", default=1, type=float, help="Maximum value of Rho allowed.")
    parser.add_argument("--rho_min", default=1e-2, type=float, help="Minimum value of Rho allowed")
    parser.add_argument("--beh_min", default=1e-2, type=float, help="Minimum value of behavior policy allowed")
    parser.add_argument("--epsilon", default=0.2, type=float, help="epsilon exploration")
    parser.add_argument("--logdir", default="logs/BPI", type=str, help="Directory for logging")
    parser.add_argument("--n_samples_td", default=10, type=int, help="num of samples for TD update")
    parser.add_argument("--n_eps_MC", default=100, type=int, help="num episodes for MC evaluation")
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

    # target policy
    target_policy = get_target_policy(num_states)

    # true value function of target_pi
    vf = mdp.get_v(target_policy)
    target_sample_means = np.array(vf)

    # target policy state distribution
    mu_s = mdp.get_state_distribution_of_policy(target_policy)
    mu_s = mu_s / np.sum(mu_s)

    #### Behavior
    behavior_policy = np.copy(target_policy)

    rho = target_policy / behavior_policy
    rho = np.clip(rho, args.rho_min, args.rho_max)
    total_env_steps = 0

    # csv writer
    header = ["num_samples", "Target(mean)", "Var(mean)", "Bias(mean)", "MSE(mean)",
              "Target(target_dist)", "Var(target_dist)", "Bias(target_dist)", "MSE(target_dist)",
              "Target(init_dist)", "Var(init_dist)", "Bias(init_dist)", "MSE(init_dist)"]
    csv_file, csv_writer = get_csv_logger(args.logdir)
    csv_writer.writerow(header)  # write header
    csv_writer.writerow([0.0] * len(header))

    # get TD object
    td = TD(env=env,
            epsilon=args.epsilon,
            max_steps=args.timeout,
            num_episodes=args.num_episodes,
            S=num_states,
            A=num_actions,
            P=mdp.P,
            R=mdp.R,
            target_pol=target_policy,
            gamma=gamma,
            alpha=alpha,
            rho_min=args.rho_min,
            rho_max=args.rho_max)
    # behavior_cor_return = [[0.0] for i in range(num_states)]
    state_values = [[] for _ in range(num_states)]

    # behavior_return_values = [[0.0] for i in range(num_states)]

    for iter_ in range(num_iter):
        metrics = {}


        if iter_ % 10 ==0 and iter_!=0:
            # get value estimate from the learnt value function
            val_target = np.array([td.get_val_s(s) for s in range(num_states)]) # V_\pi(s)
            # print("v:", val_target)
            for s in range(num_states):
                state_values[s].append(val_target[s])

            # behavior_cor_return, behavior_return_values, _ = evaluate_montecarlo_correct(env, num_actions,
            #                                                                                args.timeout,
            #                                                                                args.n_eps_MC,
            #                                                                                behavior_policy, rho,
            #                                                                                gamma,
            #                                                                                behavior_cor_return,
            #                                                                                behavior_return_values)

            # state_values = add_value_function(state_values, behavior_cor_return)

            # print("state values at iter", iter_, "is" , state_values)
            # compute metrics
            variances = [np.var(x) for x in state_values]  # var in value functions
            # print("variance:", variances)
            biases = [np.mean(x) for x in state_values]  # var in value functions
            biases = np.abs(np.array(target_sample_means) - np.array(biases))
            MSE = [np.mean(np.square(state_values[s] - target_sample_means[s])) for s in range(num_states)]
            # v_s_beh = [np.mean(x) if len(x) > 0 else 0 for x in behavior_return_values]

            #
            #
            # variances = [np.var(x) if len(x) > 0 else 0 for x in behavior_state_values]
            # v_s_beh = [np.mean(x) if len(x) > 0 else 0 for x in behavior_return_values]
            # behavior_sample_means = [np.mean(x) if len(x) > 0 else 0 for x in behavior_state_values]
            # biases = target_sample_means - np.array(behavior_sample_means)
            # MSE = np.square(biases) + variances
            # print("variances:", variances)
            # print("bias:", biases)

            # metrics["Val_Beh"] = np.mean(np.array(v_s_beh))
            metrics["num_samples"] = total_env_steps
            # var = \sum_s var(s)/|S|
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
            # print("data row:", data_row)
            csv_writer.writerow(data_row)

            # wandb logging
            wandb.log(metrics)
            if iter_ % 50 ==0:
                csv_file.flush()

        # Var and Value update
        total_env_steps += td.update(behavior_policy)

        # Policy Update
        behavior_policy = np.sqrt(np.square(target_policy) * (td.get_var() + 1e-4))
        behavior_policy = np.clip(behavior_policy, args.beh_min, 1.0)
        behavior_policy = behavior_policy / np.linalg.norm(behavior_policy, ord=1, keepdims=True, axis=1)

        if iter_ % 1000 == 0:
            # log beh policy
            np.save(os.path.join(args.logdir, "beh_" + str(total_env_steps) + ".npy"), behavior_policy)

        # # Logging
        # value_functions.append(value_function)
        # var_functions.append(var_function)
        # behavior_policies.append(np.copy(behavior_policy))
        # bp_dists.append(get_policy_distribution(env, num_states, 200, behavior_policy))
        # tgt_dists.append(target_policy_vis_dist)

    wandb.finish()
    csv_file.close()
    np.save(os.path.join(args.logdir, "final_beh_" + str(total_env_steps) + ".npy"), behavior_policy)
