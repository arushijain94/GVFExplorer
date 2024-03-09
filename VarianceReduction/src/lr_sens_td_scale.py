import sys
import argparse

import envs
import numpy as np
import gym
import wandb
import matplotlib.pyplot as plt
import csv
import src.common as common
from collections import defaultdict
from src.common import *
from src.common import LinearScheduleClass, plot_mat_single_dual_goal,get_mean_reward_scaled_exp,get_scaled_up_target_policies
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

class TD:
    def __init__(self, mdp, max_steps, num_episodes, S, A, P, size, target_pol, gamma, n_tar_policies, type_target,
                 exploration_steps, num_reward_obj, max_alpha, min_alpha, lr_decay_steps):
        self.mdp = mdp
        self.env = mdp.env
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.S = S
        self.A = A
        self.P = P
        self.size = size
        self.max_alpha = max_alpha  # learning rate
        self.min_alpha = min_alpha
        self.lr_decay_steps = lr_decay_steps
        self.gamma = gamma
        self.target_pol = target_pol
        self.n_tar_policies = n_tar_policies
        self.type_target = type_target
        self.update_counter = 0
        self.env_name = mdp.name
        self.extra_reward_info = 1
        self.goal_count_array = np.zeros(4)
        self.num_reward_obj = num_reward_obj
        self.epsilon_schedule = LinearScheduleClass(exploration_steps,
                                                    initial_p=1.0,
                                                    final_p=0.)
        self.lr_schedule = LinearScheduleClass(self.lr_decay_steps,
                                               initial_p=self.max_alpha,
                                               final_p=self.min_alpha)
        self.R_dict = get_mean_reward_scaled_exp()
        self.R = self.build_R() # [n pol, n reward objects, s, a]

        self.val_func = np.ones((self.n_tar_policies, self.num_reward_obj, self.S, self.A))  # Q(s,a) func

    def build_R(self):
        R = np.zeros((self.n_tar_policies, self.num_reward_obj, self.S, self.A))
        states_list = list(self.R_dict.keys())
        for obj_ind in range(self.num_reward_obj):
            curr_R = np.zeros((self.S, 4))
            r_state_ind = states_list[obj_ind]
            curr_R[r_state_ind, :] = self.R_dict[r_state_ind]
            R[:, obj_ind] = curr_R
        return R

    def update_val_func(self, s_t, a_t, s_tp1):
        # expected sarsa
        # v = [n pol, n rewards, s,a]
        delta = self.R[:,:, s_t, a_t] + + self.gamma * \
                np.einsum('na,nra->nr', self.target_pol[:,s_tp1,:], self.val_func[:,:,s_tp1,:]) - self.val_func[:,:, s_t, a_t]
        self.val_func[:,:, s_t, a_t] += self.curr_alpha * delta

    def update_val(self, s_a_s_next_r):
        for i in range(len(s_a_s_next_r)):
            s, a, s_next = s_a_s_next_r[i]
            self.update_val_func(s, a, s_next)


    def choose_action(self, epsilon, s, pol):
        if np.random.random() <= epsilon:
            a = np.random.choice(self.A)
        else:
            a = np.random.choice(self.A, p=pol[int(s)])
        return a

    def plot_true_value(self, num_steps, val):
        var_mat = val[:, :-1]  # [n s]
        var_mat = var_mat.reshape((self.n_tar_policies, self.size, self.size))
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, var_mat[i], self.size, num_steps, f"TRvalue_{i}")
            wandb.log({"num_samples": num_steps,
                       f"TRValue_{i}": wandb.Image(plt)})
            plt.close()
        plt = plot_mat_single_dual_goal(self.env_name,np.mean(var_mat, axis=0),  self.size, num_steps, "MeanTRValue")
        wandb.log({"num_samples": num_steps,
                   f"MeanTRValue": wandb.Image(plt)})
        plt.close()

    def plot_value(self, num_steps, log_scale=False):
        val_mat = np.mean(self.get_val(),axis=-1)[:,:-1] # [n s]
        val_mat = val_mat.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            val_mat = np.log(val_mat + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, val_mat[i], self.size, num_steps, f"value_{i}"+x)
            wandb.log({"num_samples": num_steps,
                       f"Value_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt = plot_mat_single_dual_goal(self.env_name, np.mean(val_mat, axis=0), self.size, num_steps, "MeanValue"+x)
        wandb.log({"num_samples": num_steps,
                   f"MeanValue"+x: wandb.Image(plt)})
        plt.close()

    def plot_value_diff(self, num_steps, true_val, log_scale=False):
        true_val = true_val[:, :-1]
        var_mat = np.mean(self.get_val(), axis=-1)[:, :-1]  # [n s]
        abs_diff = np.abs(true_val - var_mat)
        abs_diff = abs_diff.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            abs_diff = np.log(abs_diff + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt =  plot_mat_single_dual_goal(self.env_name, abs_diff[i], self.size, num_steps, f"diff_value_{i}"+x)
            wandb.log({"num_samples": num_steps,
                       f"diff_value_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt =  plot_mat_single_dual_goal(self.env_name, np.mean(abs_diff, axis=0), self.size, num_steps, "Mean_diff_value"+x)
        wandb.log({"num_samples": num_steps,
                   f"Mean_diff_value"+x: wandb.Image(plt)})
        plt.close()

    def get_policy_distribution(self, num_episodes, policy):
        vis_dist = np.zeros(self.S)

        for _ in range(num_episodes):
            obs = self.env.reset()
            state = np.argmax(obs)
            done = False
            step = 0
            while not done and step < self.max_steps:
                action = self.choose_action(0., state, policy)
                vis_dist[state] += 1
                obs, _, done, info = self.env.step(action)
                state = np.argmax(obs)
                step += 1
        normalized_freq = (vis_dist - np.min(vis_dist)) / (np.max(vis_dist) - np.min(vis_dist))
        return normalized_freq

    def state_to_coordinate(self, state):
        return state // self.size, state % self.size

    def plot_state_visitation(self, num_episodes, policy, num_steps):
        state_visitation = self.get_policy_distribution(num_episodes, policy)
        state_visitation = state_visitation[:-1]
        state_visitation = state_visitation.reshape((self.size, self.size))
        return plot_mat_single_dual_goal(self.env_name, state_visitation, self.size, num_steps, 'Visitation Frequency')


    def ensure_policy_is_distribution(self, policy):
        new_policy = np.clip(policy, 0., 1.0)
        new_policy = new_policy / np.linalg.norm(new_policy, ord=1, keepdims=True, axis=1)
        return new_policy

    def get_sampling_policy(self):
        if self.type_target == 1:  # mixture policy as the sampling policy
            policy = np.mean(self.target_pol, axis=0)
            # normalize policy
            policy = policy / np.linalg.norm(policy, ord=1, keepdims=True, axis=1)
        elif self.type_target == 2:  # random policy as the sampling policy
            policy = np.random.uniform(size=(num_states, num_actions))
            policy = policy/np.linalg.norm(policy, ord=1, keepdims=True, axis=1)
        else:  # round robin (1 policy each episode, use to update V value for all using off-policy)
            policy = self.target_pol[int(self.update_counter % self.n_tar_policies)]
        policy = self.ensure_policy_is_distribution(policy)
        return policy


    def update(self, total_steps):
        step = 0
        epsilon = self.epsilon_schedule.value(total_steps)
        self.curr_alpha = self.lr_schedule.value(total_steps)
        for e in range(self.num_episodes):
            s_a_s_next_r_list = []
            policy = self.get_sampling_policy()
            obs = self.env.reset()
            s = int(np.argmax(obs))
            done = False
            env_steps = 0
            while not done and env_steps < self.max_steps:
                env_steps += 1
                a = self.choose_action(epsilon, s, policy)
                obs_next, _, done, info = env.step(a)
                s_next = np.argmax(obs_next)
                step += 1
                s_a_s_next_r = (int(s), int(a), int(s_next))
                s_a_s_next_r_list.append(s_a_s_next_r)
                s = int(s_next)
            self.update_counter += 1
            self.update_val(s_a_s_next_r_list)
        return step

    def get_val(self):
        return self.val_func

    def get_V(self):
        return np.einsum("nsa,nrsa->nrs", self.target_pol, self.val_func)


def get_csv_logger(dir):
    csv_file = open(os.path.join(dir, "log.txt"), mode='w')
    csv_writer = csv.writer(csv_file)
    return csv_file, csv_writer

def add_value_function(state_values, state_returns):
    for i in range(len(state_values)):
        state_values[i].append(np.mean(state_returns[i]))  # adding avg. return to value function list
    return state_values

def get_true_vals(mdp, n_target_pol, S, target_policies, R, gamma):
    # get true V_pi
    true_vals = np.ones((n_target_pol, S))
    for i in range(n_target_pol):
        true_vals[i] = np.array(mdp.get_v(R[i], gamma, target_policies[i]))
    return true_vals

def get_true_val_scaled_exp(mdp, n_target_pol, num_reward_objs, S, R, gamma):
    # R = [n pol, n reward obj, s,a]
    true_vals=np.ones((n_target_pol, num_reward_objs, S))
    for pol_ind in range(n_target_pol):
        for obj_ind in range(num_reward_objs):
            true_vals[pol_ind,obj_ind] = np.array(mdp.get_v(R[pol_ind,obj_ind], gamma, target_policies[pol_ind]))
    return true_vals # [pols, objs, s]

def compute_mean_non_wall_states(mdp, values):
    S = mdp.non_wall_states
    num_states = len(S)
    num_policies = values.shape[0]
    mean_values = np.zeros(num_policies)
    for i in range(num_policies):
        for s in S:
            mean_values[i] += values[i, s]
        mean_values[i] /= num_states
    return mean_values




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes for tuning PI")
    parser.add_argument("--num_iter", default=20000, type=int,
                        help="Total number of iterations of updating the behaviors policy.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="Grid_5_Online_SingleGoal", type=str, help="Group name for Wandb")
    parser.add_argument("--wandb_name", default="Det_10_Target_TD", type=str, help="Log name for Wandb")
    parser.add_argument("--timeout", default=500, type=int, help="Max episode length of the environment.")
    parser.add_argument("--size", default=20, type=int, help="Side dimension of the grid.")
    parser.add_argument("--logdir", default="/home/mila/j/jainarus/scratch/VarReduction/Det_10/Feb_20/try", type=str, help="Directory for logging")
    parser.add_argument("--stochastic_env", default=True, type=bool, help="Type of transition F: det; T:stochastic")
    parser.add_argument("--max_alpha", default=1.0, type=float, help="Learning Rate")
    parser.add_argument("--min_alpha", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay_steps", default=500000, type=int, help="Learning Rate decay in these many steps")
    parser.add_argument("--n_target_pol", default=8, type=int, help="Num of Target Policies")
    parser.add_argument("--type_target", default=0, type=int, help="(0: roundrobin, 1: mixture policy, 2: random policy)")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--env_name", default="grid_scaled_up_exp", type=str, help="{grid_scaled_up_exp}")
    parser.add_argument("--last_k", default=500, type=int, help="ast k values for moving reward func")
    parser.add_argument("--exploration_steps", default=500, type=int, help="exploration steps")
    parser.add_argument("--num_reward_obj", default=16, type=int, help="num of diff reward fucntions")

    # we want to evaluation num_target_pol * num_reward_obj V values for all states

    args = parser.parse_args()
    np.random.seed(args.seed)
    last_k = args.last_k
    num_reward_obj = args.num_reward_obj

    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_dir = f"{args.wandb_group}_{current_time}_{args.min_alpha}_{args.lr_decay_steps}_{args.seed}"
    wandb_dir = f"/home/mila/j/jainarus/scratch/VarReduction/wandb/{wandb_dir}"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    wandb.init(project="simple-gvf-evaluations", entity="jainarus", group=args.wandb_group,
               config=namespace_to_dict(args),
               name="seed_" + str(args.seed),
               dir=wandb_dir,
               mode="online")

    # logging data
    log_dir = args.logdir  # os.path.join(args.logdir, args.wandb_group, args.wandb_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # deterministic 5X5 MDP
    mdp = common.create_env(args.env_name, size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    env = mdp.env
    gamma = mdp.gamma

    num_iter = args.num_iter
    num_episodes = args.num_episodes
    n_target_pol = args.n_target_pol
    type_target = args.type_target
    env_name= args.env_name


    target_policies = get_scaled_up_target_policies(num_states, args.env_name, args.n_target_pol)

    min_alpha = args.min_alpha
    lr_q_list = [0.1, 0.25, 0.5, 0.8, 0.9, 0.95]
    mse_value_list = []
    samples_consumed = []

    for i in range(len(lr_q_list)):
        min_alpha=lr_q_list[i]
        total_env_steps = 0
        # get TD object
        td = TD(mdp=mdp,
                max_steps=args.timeout,
                num_episodes=args.num_episodes,
                S=num_states,
                A=num_actions,
                P=mdp.P,
                size=args.size,
                target_pol=target_policies,
                gamma=gamma,
                max_alpha=args.max_alpha,
                min_alpha=min_alpha,
                lr_decay_steps=args.lr_decay_steps,
                n_tar_policies=n_target_pol,
                type_target=type_target,
                exploration_steps=args.exploration_steps,
                num_reward_obj=num_reward_obj
                )

        true_vals = get_true_val_scaled_exp(mdp, n_target_pol, num_reward_obj, num_states, td.R, gamma) # [n pols, n reward objs, s]

        estimated_target = []
        for iter_ in range(num_iter):
            total_env_steps += td.update(total_env_steps)
            val_target = td.get_V()

            if total_env_steps>500000:
                metrics={}
                mse_val= np.mean(np.square(true_vals - val_target))
                mse_value_list.append(mse_val)
                samples_consumed.append(total_env_steps)
                metrics["LR_Q"] = min_alpha
                metrics["mse_avg"] = mse_val
                metrics["num_samples"] = total_env_steps
                wandb.log(metrics)
                print(f" here ----- alpha: {min_alpha} and avg_mse: {mse_val} at samples: {total_env_steps}")
                break

    for i in range(len(lr_q_list)):
        print(f"alpha: {lr_q_list[i]} and avg_mse: {mse_value_list[i]} at samples: {samples_consumed[i]}")

    wandb.finish()
