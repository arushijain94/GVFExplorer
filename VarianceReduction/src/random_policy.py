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
from src.common import namespace_to_dict, create_multiple_reward, get_mult_simple_target_policy
from collections import defaultdict


class TD:
    def __init__(self, env, max_steps, num_episodes, S, A, P, R, target_pol, gamma, alpha, n_tar_policies, type_target,
                 val_func=None):
        self.env = env
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.target_pol = target_pol
        self.n_tar_policies = n_tar_policies
        self.val_func = val_func
        self.type_target = type_target
        self.update_counter = 0
        if val_func == None:
            self.val_func = np.zeros((self.n_tar_policies, self.S, self.A))  # Q(s,a) func

    def update_val_func(self, s_t, a_t, s_tp1, policy):
        # expected sarsa
        delta = self.R[:,s_t, a_t] + self.gamma * self.get_val_s(s_tp1, policy) - self.val_func[:,s_t, a_t] # n dim vector for delta
        self.val_func[:,s_t, a_t] += self.alpha * delta

    def update_val(self, s_a_s_next, policy):
        for i in range(len(s_a_s_next)):
            s, a, s_next = s_a_s_next[i]
            self.update_val_func(s, a, s_next, policy)

    def get_val_s(self, state, policy):
        val = np.einsum('na,na->n', self.target_pol[:, state], self.val_func[:, state, :])
        #
        # rho = np.array([self.target_pol[i]/policy for i in range(self.n_tar_policies)])
        # rho = np.clip(rho, 1e-3, 1.)
        # val = np.einsum('na,na->n', rho[:,state,:], self.val_func[:,state,:])
        # print("val:", val)
        # v(s) = \sum_a \rho(s,a) q(s,a)
        return val

    def get_sampling_policy(self):
        if self.type_target == 1:  # mixture policy as the sampling policy
            policy = np.mean(self.target_pol, axis=0)
            # normalize policy
            policy = policy / np.linalg.norm(policy, ord=1, keepdims=True, axis=1)
            return policy
        else:  # round robin (1 policy each episode, use to update V value for all using off-policy)
            return self.target_pol[int(self.update_counter % self.n_tar_policies)]


    def update(self):
        step = 0
        for e in range(self.num_episodes):
            s_a_s_next_list = []
            policy = self.get_sampling_policy()
            obs = self.env.reset()
            s = int(np.argmax(obs))
            done = False
            env_steps = 0
            while not done and env_steps < self.max_steps:
                env_steps += 1
                a = np.random.choice(self.A, p=policy[s])
                obs_next, _, done, info = env.step(a)
                s_next = np.argmax(obs_next)
                step += 1
                s_a_s_next = (int(s), int(a), int(s_next))
                s_a_s_next_list.append(s_a_s_next)
                s = int(s_next)
            self.update_counter += 1
            self.update_val(s_a_s_next_list, policy)
        # print("val:", self.val_func)
        return step

    def get_val(self):
        return self.val_func


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
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--n_target_pol", default=3, type=int, help="Num of Target Policies")
    parser.add_argument("--tar_pol_loc", default="/home/mila/j/jainarus/scratch/VarReduction/Det_10/target_pols",
                        type=str, help="Loc of Target Policies")
    parser.add_argument("--type_target", default=0, type=int, help="(0: roundrobin, 1: mixture policy)")
    parser.add_argument("--seed", default=1, type=int, help="seed")


    args = parser.parse_args()
    np.random.seed(args.seed)

    wandb.init(project="BPG", entity="jainarus", group=args.wandb_group,
               name=args.wandb_name, config=namespace_to_dict(args),
               mode="online")

    # logging data
    log_dir = args.logdir  # os.path.join(args.logdir, args.wandb_group, args.wandb_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # deterministic 5X5 MDP
    mdp = TabularMDP(size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    env = mdp.env
    alpha = args.alpha

    num_iter = args.num_iter
    num_episodes = args.num_episodes
    gamma = args.gamma
    n_target_pol = args.n_target_pol
    type_target = args.type_target

    # load target policies
    # target_policies = []
    # for pol_ind in range(args.n_target_pol):
    #     target_policies.append(np.load(os.path.join(args.tar_pol_loc, 'pol_' + str(pol_ind + 1) + '.npy')))
    # target_policies = np.array(target_policies)

    # simple target policies
    target_policies = get_mult_simple_target_policy(num_states, num_actions)

    # true value function of target_pi
    true_vals = np.ones((n_target_pol, num_states))
    rho = np.ones((n_target_pol, num_states, num_actions))
    for i in range(n_target_pol):
        true_vals[i] = np.array(mdp.get_v(target_policies[i]))

    state_values = defaultdict(list)  # (target_ind, num_states ind)

    total_env_steps = 0

    # get TD object
    td = TD(env=env,
            max_steps=args.timeout,
            num_episodes=args.num_episodes,
            S=num_states,
            A=num_actions,
            P=mdp.P,
            R=create_multiple_reward(n_target_pol, args.size, num_states, num_actions),
            target_pol=target_policies,
            gamma=gamma,
            alpha=alpha,
            n_tar_policies=n_target_pol,
            type_target=type_target, )


    for iter_ in range(num_iter):
        metrics = {}
        total_env_steps += td.update()
        if iter_ % 1 == 0:
            val_target = td.get_val()
            for tar_ind in range(n_target_pol):
                for s in range(num_states):
                    state_values[(tar_ind, s)].append(val_target[tar_ind, s])

        if iter_ % 3 == 0 and iter_ != 0:
            metrics={}
            vars = np.zeros((n_target_pol, num_states))
            bias = np.zeros((n_target_pol, num_states))
            mse = np.zeros((n_target_pol, num_states))
            for tar_ind in range(n_target_pol):
                for s in range(num_states):
                    vars[tar_ind, s] = np.var(state_values[(tar_ind, s)])
                    bias[tar_ind, s] = np.abs(true_vals[tar_ind, s] - np.mean(state_values[(tar_ind, s)]))
                    mse[tar_ind, s] = np.mean(np.square(true_vals[tar_ind, s] - state_values[(tar_ind, s)]))

            # n dim metrics
            var_metrics = np.mean(vars, axis=1)  # n_dim
            bias_metrics = np.mean(bias, axis=1)
            mse_metrics = np.mean(mse, axis=1)

            for i in range(n_target_pol):
                metrics["var_" + str(i + 1)] = var_metrics[i]
                metrics["bias_" + str(i + 1)] = bias_metrics[i]
                metrics["mse_" + str(i + 1)] = mse_metrics[i]
                metrics["trueval_" + str(i + 1)] = np.mean(true_vals[i])

            metrics["var_avg"] = np.mean(var_metrics)
            metrics["bias_avg"] = np.mean(bias_metrics)
            metrics["mse_avg"] = np.mean(mse_metrics)
            metrics["true_target_val"] = np.mean(true_vals)
            metrics["num_samples"] = total_env_steps

            # wandb logging
            wandb.log(metrics)

    wandb.finish()
