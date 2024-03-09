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
from src.common import namespace_to_dict, get_target_policy

class TD:
    def __init__(self, env, max_steps, num_episodes, S, A, P, R, target_pol, gamma, alpha, val_func=None):
        self.env = env
        self.max_steps=max_steps
        self.num_episodes = num_episodes
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.target_pol = target_pol
        self.val_func = val_func
        if val_func == None:
            self.val_func = np.zeros((self.S, self.A))  # Q(s,a) func

    def update_val_func(self, s_t, a_t, s_tp1):
        # expected sarsa
        delta = self.R[s_t, a_t] + self.gamma * self.get_val_s(s_tp1) - self.val_func[s_t, a_t]
        self.val_func[s_t, a_t] += self.alpha * delta


    def update_val(self, s_a_s_next):
        for i in range(len(s_a_s_next)):
            s, a, s_next = s_a_s_next[i]
            self.update_val_func(s, a, s_next)


    def get_val_s(self, state):
        # v(s) = \sum_a \pi(s,a) q(s,a)
        return np.sum(self.target_pol[state] * self.val_func[state, :])


    def update(self):
        step = 0
        s_a_s_next_list = []
        for e in range(self.num_episodes):
            obs = self.env.reset()
            s= np.argmax(obs)
            done = False
            env_steps=0
            while not done and env_steps< self.max_steps:
                env_steps+=1
                a = np.random.choice(self.A, p=self.target_pol[int(s)])
                obs_next, r, done, info = env.step(a)
                s_next = np.argmax(obs_next)
                step +=1
                s_a_s_next = (int(s),int(a),int(s_next))
                s_a_s_next_list.append(s_a_s_next)
                s = s_next
        self.update_val(s_a_s_next_list)
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
    alpha = args.alpha

    num_iter = args.num_iter
    num_episodes = args.num_episodes
    gamma = args.gamma

    # target policy
    target_pi = get_target_policy(num_states)

    # true value function of target_pi
    vf = mdp.get_v(target_pi)
    target_sample_means = vf

    # target policy state distribution
    mu_s = mdp.get_state_distribution_of_policy(target_pi)
    mu_s = mu_s / np.sum(mu_s)


    # maintains the list G(s) seen so far after every evaluation episode
    state_return = [[] for _ in range(num_states)]
    total_env_steps = 0

    # maintains the list V(s) of all the value functions seen so far
    state_values = [[] for _ in range(num_states)]

    header = ["num_samples", "Target(mean)","Var(mean)", "Bias(mean)", "MSE(mean)",
              "Target(target_dist)", "Var(target_dist)", "Bias(target_dist)","MSE(target_dist)",
              "Target(init_dist)",  "Var(init_dist)", "Bias(init_dist)", "MSE(init_dist)"]

    csv_file, csv_writer = get_csv_logger(log_dir)
    csv_writer.writerow(header)  # write header
    csv_writer.writerow([0.0] * len(header))

    # get TD object
    td = TD(env=env,
            max_steps=args.timeout,
            num_episodes=args.num_episodes,
            S=num_states,
            A=num_actions,
            P=mdp.P,
            R=mdp.R,
            target_pol=target_pi,
            gamma=gamma,
            alpha=alpha,)

    state_values = [[] for _ in range(num_states)]

    for iter_ in range(num_iter):
        metrics = {}
        total_env_steps += td.update()
        if iter_ % 5 ==0:
            val_target = np.array([td.get_val_s(s) for s in range(num_states)])  # V_\pi(s)
            for s in range(num_states):
                state_values[s].append(val_target[s])


        # # Value Update
        # state_return, new_steps = evaluate_montecarlo_correct(env, num_states, args.timeout, num_episodes, target_pi,
        #                                                       np.ones((num_states, num_actions)), gamma, state_return)
        # total_env_steps += new_steps
        #
        # state_values = add_value_function(state_values, state_return)


        if iter_ % 10 ==0 and iter_!=0:

            # compute metrics
            variances = [np.var(x) for x in state_values] # var in value functions
            biases = [np.mean(x) for x in state_values]  # var in value functions
            biases = np.abs(np.array(target_sample_means) - np.array(biases))
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
            if iter_ % 100 == 0:
                csv_file.flush()

    wandb.finish()
    csv_file.close()
