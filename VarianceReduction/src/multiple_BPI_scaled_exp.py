### Q:
# 1. Correct for MC update of the vale functions using multiple rollouts
# mu -> x rollouts -> off-policy TD -> get update v everytime because task is changing, so true v changes and estimates would also change.
# import os
# import sys
import argparse
# import random
# import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from collections import defaultdict
import src.common as common
from src.common import *
from src.common import LinearScheduleClass, plot_mat_single_dual_goal,get_mean_reward_scaled_exp,get_scaled_up_target_policies
import numpy as np
# import seaborn as sns
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
class TD:
    def __init__(self, mdp, max_steps, num_episodes, S, A, P, size, target_pol, gamma, rho_min, rho_max,
                 n_tar_policies, exploration_steps, temperature_steps, type_behavior_policy, exp_next_val,
                 num_reward_obj,max_alpha_Q, max_alpha_var, min_alpha_Q,
                 lr_decay_steps_Q, lr_decay_steps_var, min_alpha_var,
                 online_alpha=0.05):
        self.mdp = mdp
        self.env = mdp.env
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.env_name = mdp.name
        self.S = S
        self.A = A
        self.P = P
        self.size = size  # n dim R matrix for multiple policies
        self.rho_min = rho_min
        self.rho_max = rho_max

        self.max_alpha_Q = max_alpha_Q  # learning rate
        self.max_alpha_var = max_alpha_var  # learning rate
        self.min_alpha_Q = min_alpha_Q
        self.lr_decay_steps_Q = lr_decay_steps_Q
        self.lr_decay_steps_var = lr_decay_steps_var
        self.min_alpha_var = min_alpha_var
        self.curr_alpha_Q = max_alpha_Q  # this change with every step
        self.curr_alpha_var = max_alpha_var  # this change with every step

        self.gamma = gamma
        self.target_pol = target_pol
        self.n_tar_policies = n_tar_policies
        self.num_reward_obj = num_reward_obj
        self.val_func = np.ones((self.n_tar_policies, self.num_reward_obj, self.S, self.A))*2  # Q(s,a) func
        self.var_func = np.ones((self.n_tar_policies, self.num_reward_obj, self.S, self.A))*0.5
        self.R_dict = get_mean_reward_scaled_exp()
        self.R = self.build_R()  # [n pol, n reward objects, s, a]


        self.online_mean = np.zeros((self.n_tar_policies, self.S, self.A))
        self.online_var = np.ones((self.n_tar_policies, self.S, self.A))
        self.online_alpha = online_alpha
        self.extra_reward_info = 1
        self.var_max = np.ones((self.n_tar_policies, self.S, self.A))*0.01
        self.goal_count_array = np.zeros(4)
        self.type_behavior_policy = type_behavior_policy
        self.exp_next_val = exp_next_val

        self.s_a_list = self.get_s_a_pairs()
        self.len_s_a_list = len(self.s_a_list)
        self.epsilon_schedule = LinearScheduleClass(exploration_steps,
                                                    initial_p=1.0,
                                                    final_p=0.)
        self.temp_schedule = LinearScheduleClass(temperature_steps,
                                                    initial_p=1.0,
                                                    final_p=0.1)
        # q LR
        self.lr_schedule_Q = LinearScheduleClass(self.lr_decay_steps_Q,
                                                 initial_p=self.max_alpha_Q,
                                                 final_p=self.min_alpha_Q)  # linearly decay lr with every step
        # var lr
        self.lr_schedule_var = LinearScheduleClass(self.lr_decay_steps_var,
                                                   initial_p=self.max_alpha_var,
                                                   final_p=self.min_alpha_var)
        self.temp = 1.0

    def build_R(self):
        R = np.zeros((self.n_tar_policies, self.num_reward_obj, self.S, self.A))
        states_list = list(self.R_dict.keys())
        for obj_ind in range(self.num_reward_obj):
            curr_R = np.zeros((self.S, 4))
            r_state_ind = states_list[obj_ind]
            curr_R[r_state_ind, :] = self.R_dict[r_state_ind]
            R[:, obj_ind] = curr_R
        return R


    def get_s_a_pairs(self):
        return list(itertools.product(range(self.S), range(self.A)))

    def set_beh_policy(self, policy):
        self.behavior_pol = policy
        # self.rho = np.array([set_rho(self.target_pol[i], self.behavior_pol, self.rho_min, self.rho_max) for i in range(self.n_tar_policies)])
        # print("rho:", self.rho)


    def update_val_func(self, s_t, a_t, s_tp1):
        # expected sarsa
        delta = self.R[:, :, s_t, a_t] + + self.gamma * \
                np.einsum('na,nra->nr', self.target_pol[:, s_tp1, :], self.val_func[:, :, s_tp1, :]) \
                - self.val_func[:,:, s_t, a_t]
        self.val_func[:, :, s_t, a_t] += self.curr_alpha_Q * delta

    def update_val(self, s_a_s_next_r):
        for i in range(len(s_a_s_next_r)):
            s, a, s_next = s_a_s_next_r[i]
            self.update_val_func(s, a, s_next)

    def get_var_s(self, state):
        if self.exp_next_val:
            return np.einsum('na,na->n', self.target_pol[:, state, :], self.var_func[:, state, :])
        else:
            return np.einsum('a,na, na->n', self.behavior_pol[state, :], self.rho[:, state, :] ** 2,
                             self.var_func[:, state, :])
        # var(s) = \sum_a \mu(s,a) \rho^2(s,a) var(s,a)
        # return np.einsum('a,na, na->n', self.behavior_pol[state, :], self.rho[:, state, :] ** 2,
        #                  self.var_func[:, state, :])
        # return np.sum(self.behavior_pol[state] * (self.rho[state, :] ** 2) * self.var_func[state, :])
        # var(s) = \sum \pi(s,a) var(s,a)
        # return np.einsum('na,na->n', self.target_pol[:, state, :], self.var_func[:, state, :])

    def get_val_s(self, state):
        if self.exp_next_val:
            return np.einsum('na,na->n', self.target_pol[:, state, :], self.val_func[:, state, :])
        else:
            return np.einsum('a,na, na->n', self.behavior_pol[state, :], self.rho[:, state, :],
                             self.val_func[:, state, :])

        # v(s) = \sum_a \pi(s,a) q(s,a)
        # return np.einsum('na,na->n', self.target_pol[:, state, :], self.val_func[:, state, :])
        # return np.sum(self.target_pol[state] * self.val_func[state, :])

    def update_var_func(self, s_t, a_t, s_tp1):
        # s_tp1 = np.random.choice(self.S, p=self.P[s_t, a_t])
        delta_v = self.R[:, :, s_t, a_t] + self.gamma * \
                np.einsum('na,nra->nr', self.target_pol[:, s_tp1, :], self.val_func[:, :, s_tp1, :]) \
                - self.val_func[:, :, s_t, a_t] # [n,r] dim

        delta_var = delta_v**2 + self.gamma**2 *\
                    np.einsum('na,nra->nr',self.target_pol[:,s_tp1,:], self.var_func[:,:,s_tp1,:])\
                    - self.var_func[:,:,s_t,a_t] # [n,r] dim
        self.var_func[:, :, s_t, a_t] += self.curr_alpha_var* delta_var

    def update_var(self, s_a_s_next_r):
        for i in range(len(s_a_s_next_r)):
            s, a, s_next = s_a_s_next_r[i]
            self.update_var_func(s, a, s_next)

    def choose_beh_action(self, epsilon, temp, s, pol):
        if np.random.random() <= epsilon:
            a = np.random.choice(self.A)
        else:
            a = np.random.choice(self.A, p=pol[int(s)])
        return a

    def update(self, behavior_policy, total_steps):
        self.set_beh_policy(behavior_policy)
        step = 0
        s_a_s_next_r_list = []
        epsilon = self.epsilon_schedule.value(total_steps)
        self.temp = self.temp_schedule.value(total_steps)
        self.curr_alpha_Q = self.lr_schedule_Q.value(total_steps)
        self.curr_alpha_var = self.lr_schedule_var.value(total_steps)
        for e in range(self.num_episodes):
            obs = self.env.reset()
            s = np.argmax(obs)
            done = False
            env_steps = 0
            while not done and env_steps < self.max_steps:
                a = self.choose_beh_action(epsilon, self.temp, s, self.behavior_pol)
                obs_next, _, done, info = self.env.step(a)
                s_next = np.argmax(obs_next)
                step += 1
                s_a_s_next_r = (int(s), int(a), int(s_next))
                s_a_s_next_r_list.append(s_a_s_next_r)
                s = s_next
                env_steps += 1
        # print(f"steps in 1 epsiode:{env_steps}")
        self.update_val(s_a_s_next_r_list)
        np.random.shuffle(s_a_s_next_r_list)
        self.update_var(s_a_s_next_r_list)
        return step, self.var_func

    def function_normalization(self):
        # online normalization via https://proceedings.neurips.cc/paper_files/paper/2019/file/cb3ce9b06932da6faaa7fc70d5b5d2f4-Paper.pdf
        # y_t = x_t - mean_{t-1}/std__{t-1}
        # mean_t = alpha*mean_{t-1} + (1-alpha)*mean(x_t)
        # var_t = alpha*var_{t-1} + alpha*(1-alpha)*[mean(x_t) - mean_{t-1}]^2
        normalized_variance = np.array([ (self.var_func[i]-self.online_mean[i])/ np.sqrt(np.abs(self.online_var[i])) for i in range(self.n_tar_policies)])
        self.online_mean = (1 - self.online_alpha) * self.online_mean + self.online_alpha* self.var_func
        self.online_var = (1 - self.online_alpha) * self.online_var + self.online_alpha * (self.var_func - self.online_mean)**2.0
        self.online_var[self.online_var<1e-4] = 1e-4
        normalized_variance[normalized_variance < 1e-4] = 1e-4
        return normalized_variance


    def update_var_max(self):
        for i in range(n_target_pol):
            self.var_max[i] = np.maximum( self.var_max[i], self.var_func[i])

    def get_var(self):
        return self.var_func

    def get_val(self):
        return self.val_func
    def get_V(self):
        return np.einsum("nsa,nrsa->nrs", self.target_pol, self.val_func)

    def get_policy_distribution(self, num_episodes, policy):
        vis_dist = np.zeros(self.S)

        for _ in range(num_episodes):
            obs = self.env.reset()
            state = np.argmax(obs)
            done = False
            step = 0
            while not done and step < self.max_steps:
                action = self.choose_beh_action(0., 0.1, state, policy)
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
        print(f"Steps: {num_steps} and state visit {state_visitation}")
        return plot_mat_single_dual_goal(self.env_name, state_visitation, self.size, num_steps, 'Visitation Frequency')

    def plot_var(self, num_steps, log_scale=False, vmin=0, vmax=4, show_v_limit=False):
        val_mat = np.mean(self.get_var(),axis=-1)[:,:-1] # [n s]
        val_mat = val_mat.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            val_mat = np.log(val_mat + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, val_mat[i], self.size, num_steps, f"var_{i}"+x, vmin=vmin, vmax=vmax, show_v_limit=show_v_limit)
            wandb.log({"num_samples": num_steps,
                       f"Var_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt = plot_mat_single_dual_goal(self.env_name, np.mean(val_mat, axis=0), self.size, num_steps, "MeanVar"+x, vmin=vmin, vmax=vmax, show_v_limit=show_v_limit)
        wandb.log({"num_samples": num_steps,
                   f"MeanVar"+x: wandb.Image(plt)})
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

    def plot_value_diff(self, num_steps, true_val, log_scale=False, vmin=0, vmax=4, show_v_limit=False):
        true_val = true_val[:, :-1]
        var_mat = np.mean(self.get_val(), axis=-1)[:, :-1]  # [n s]
        abs_diff = np.abs(true_val - var_mat)
        abs_diff = abs_diff.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            abs_diff = np.log(abs_diff + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, abs_diff[i], self.size, num_steps, f"diff_value_{i}"+x, vmin=vmin, vmax=vmax, show_v_limit=show_v_limit)
            wandb.log({"num_samples": num_steps,
                       f"diff_value_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt =  plot_mat_single_dual_goal(self.env_name, np.mean(abs_diff, axis=0), self.size, num_steps, "Mean_diff_value"+x, vmin=vmin, vmax=vmax, show_v_limit=show_v_limit)
        wandb.log({"num_samples": num_steps,
                   f"Mean_diff_value"+x: wandb.Image(plt)})
        plt.close()


    def ensure_policy_is_distribution(self, policy):
        new_policy = np.clip(policy, 0., 1.0)
        new_policy = new_policy / np.linalg.norm(new_policy, ord=1, keepdims=True, axis=1)
        return new_policy

    def compute_objective(self, var, target_policies, type_behavior_policy):
        num_policies = var.shape[0]
        tolerance = 1e-4
        temp = self.temp

        # weights_num = 1. / num_policies * np.ones_like(var)
        var_num = var + tolerance
        numerator = np.sqrt(np.einsum('nsa,nrsa->sa',
                                      np.square(target_policies), var_num))

        # if type_behavior_policy == 1:  # softmax directly on var value
        #     numerator = softmax_numerator(np.mean(var_num, axis=0), temp)
        # elif type_behavior_policy == 2:  # first one but with tolerance in target policy
        #     new_target_policies = np.clip(target_policies, 0.05, 1.0)
        #     new_target_policies = new_target_policies / np.linalg.norm(new_target_policies, ord=1, keepdims=True,
        #                                                                axis=-1)
        #     numerator = np.sqrt(np.einsum('nsa,nsa,nsa->sa', np.square(new_target_policies), weights_num, var_num))
        # else:
        #     # type_behavior_policy == 0:  # \mu(s,a) = \sqrt(\sum_i \pi^2(s,a) * var(s,a))
        #     numerator = np.sqrt(np.einsum('nsa,nrsa,nrsa->sa', np.square(target_policies), weights_num, var_num))
        return numerator, np.ones_like(var)

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

def clip_normalize_policy(behavior_policy, beh_min, target_policies):
    # sum_target_policies = np.sum(target_policies, axis=0)
    # zero_mask = (sum_target_policies ==0)
    new_behavior_policy = np.clip(behavior_policy, beh_min, 1.0)
    # new_behavior_policy[zero_mask] = 0.
    new_behavior_policy = new_behavior_policy / np.linalg.norm(new_behavior_policy, ord=1, keepdims=True, axis=1)
    return new_behavior_policy

def softmax_numerator(val, temp):
    # Scale Q-values by temperature
    scaled_values = val / temp # val [s, a]

    # Compute softmax
    # Subtract max for numerical stability (prevent overflow)
    max_q = np.max(scaled_values, axis=-1, keepdims=True)
    exp_q = np.exp(scaled_values - max_q)
    # sum_exp_q = np.sum(exp_q, axis=1, keepdims=True)
    # probabilities = exp_q / sum_exp_q
    return exp_q



def compute_var_objective(type_obj, var, norm_var, target_policies, online_alpha, ema_var, type_behavior_policy, temp):
    num_policies = var.shape[0]
    tolerance = 1e-4
    def normalized_weights_func(ema_var, var_value):
        ema_var = (1 - online_alpha) * ema_var + online_alpha * var_value
        normalized_ema_var = ema_var / np.linalg.norm(ema_var, ord=1, keepdims=True, axis=0)
        weights = 1.0 / normalized_ema_var  # weights are inversely proportional to variances
        normalized_weights = weights / np.sum(weights, axis=0)
        return normalized_weights, ema_var

    if type_obj==1: # "var_sum"
        weights_num = 1./num_policies * np.ones_like(var)
        var_num = var + tolerance
    elif type_obj== 2: #"norm_var_sum":
        weights_num = 1. / num_policies * np.ones_like(var)
        var_num = norm_var + tolerance
    elif type_obj==3: #"var_weighted_sum":
        var_num = var + tolerance
        weights_num, ema_var = normalized_weights_func(ema_var, var_num)
    elif type_obj==4: #"norm_var_weighted_sum":
        var_num = norm_var + tolerance
        weights_num, ema_var = normalized_weights_func(ema_var, var_num)
    else:
        print("wrong objective for variance --> check multiple_BPI.py!!")
        sys.exit(-1)

    if type_behavior_policy == 0: # \mu(s,a) = \sqrt(\sum_i \pi^2(s,a) * var(s,a))
        numerator = np.sqrt(np.einsum('nsa,nsa,nsa->sa', np.square(target_policies), weights_num,  var_num))
    elif type_behavior_policy == 1: # softmax directly on var value
        var = var+tolerance
        numerator = softmax_numerator(np.mean(var, axis=0), temp)
    elif type_behavior_policy == 2: # first one but with tolerance in target policy
        new_target_policies = np.clip(target_policies, 0.05, 1.0)
        new_target_policies = new_target_policies / np.linalg.norm(new_target_policies, ord=1, keepdims=True, axis=-1)
        numerator = np.sqrt(np.einsum('nsa,nsa,nsa->sa', np.square(new_target_policies), weights_num,  var_num))
    return numerator, ema_var, weights_num




if __name__ == '__main__':
    '''
    TODO:
    1. Add min clip for action from a policy
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes for tuning PI")
    parser.add_argument("--num_iter", default=10000, type=int,
                        help="Total number of iterations of updating the behaviors policy.")

    parser.add_argument("--max_alpha_Q", default=0.9, type=float, help="Learning Rate")
    parser.add_argument("--min_alpha_Q", default=0.05, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay_steps_Q", default=100000, type=int, help="Learning Rate decay in these many steps")
    parser.add_argument("--max_alpha_var", default=0.7, type=float, help="Learning Rate")
    parser.add_argument("--min_alpha_var", default=0.05, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay_steps_var", default=100000, type=int,
                        help="Learning Rate decay in these many steps")

    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="Grid_BPI_try1", type=str, help="Group name for Wandb")
    parser.add_argument("--wandb_name", default="Det_10", type=str, help="Log name for Wandb")
    parser.add_argument("--timeout", default=500, type=int, help="Max episode length of the environment.")
    parser.add_argument("--rho_max", default=1, type=float, help="Maximum value of Rho allowed.")
    parser.add_argument("--rho_min", default=1e-3, type=float, help="Minimum value of Rho allowed")
    parser.add_argument("--beh_min", default=1e-3, type=float, help="Minimum value of behavior policy allowed")
    parser.add_argument("--logdir", default="/home/mila/j/jainarus/scratch/VarReduction/Det_10/Feb_20/try", type=str, help="Directory for logging")
    parser.add_argument("--size", default=20, type=int, help="size of MDP")
    parser.add_argument("--stochastic_env", default=True, type=bool, help="Type of transition F: det; T:stochastic")
    parser.add_argument("--n_target_pol", default=8, type=int, help="Num of Target Policies")
    parser.add_argument("--random_start_beh", default=2, type=int, help="(0: one of target policy as start policy, 1: random policy, 2: mixture target policies)")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--online_alpha", default=0.1, type=float, help="alpha for normalization")
    parser.add_argument("--env_name", default="grid_scaled_up_exp", type=str, help="{simple_grid, online_rooms_same_reward_grid, online_rooms_different_reward_grid,"
                                                                                              "semigreedy_online_rooms_same_reward_grid, semigreedy_online_rooms_different_reward_grid}")
    parser.add_argument("--type_obj", default=1, type=int, help="{1:var_sum, 2:norm_var_sum, 3:var_weighted_sum, 4:norm_var_weighted_sum}")
    parser.add_argument("--last_k", default=200, type=int, help="last k values for moving reward func")
    parser.add_argument("--exploration_steps", default=500, type=int, help="exploration steps")
    parser.add_argument("--type_behavior_policy", default=0, type=int, help="(0: \pi^2 var(s,a), 1: softmax(mean var) 2: \sum_i clip(pi_i)^2 var_i)")
    parser.add_argument("--temperature_steps", default=200, type=int, help="temperature steps")
    parser.add_argument("--exp_next_val", default=1, type=int, help="(0: use rho 1: exp under pi)")
    parser.add_argument("--num_reward_obj", default=16, type=int, help="num of diff reward fucntions")

    args = parser.parse_args()
    np.random.seed(args.seed)
    last_k = args.last_k
    num_reward_obj = args.num_reward_obj

    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_dir = f"{args.wandb_group}_{current_time}_{args.min_alpha_Q}_{args.lr_decay_steps_Q}_{args.seed}"
    wandb_dir = f"/home/mila/j/jainarus/scratch/VarReduction/wandb/{wandb_dir}"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    # common.set_seed(args.seed) # seed for common program

    # initializing wandb for storing results
    wandb.init(project="simple-gvf-evaluations", entity="jainarus", group=args.wandb_group,
               config=namespace_to_dict(args),
               name="seed_"+ str(args.seed),
               dir=wandb_dir)

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # deterministic 5X5 MDP
    mdp = common.create_env(args.env_name, size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    gamma = mdp.gamma
    env = mdp.env

    exp_next_val = args.exp_next_val
    num_iter = args.num_iter
    num_episodes = args.num_episodes
    n_target_pol = args.n_target_pol
    online_alpha = args.online_alpha
    type_behavior_policy= args.type_behavior_policy

    # load target policies
    target_policies = get_scaled_up_target_policies(num_states, args.env_name, args.n_target_pol)

    # initialization of behavior policy
    if args.random_start_beh ==1:
        behavior_policy = np.random.uniform(size=(num_states, num_actions))
    elif args.random_start_beh ==0:
        ind = np.random.choice(n_target_pol)
        behavior_policy = np.copy(target_policies[ind])
    else: # mixture policy
        behavior_policy = np.mean(target_policies, axis=0)
    behavior_policy = behavior_policy / np.linalg.norm(behavior_policy, ord=1, keepdims=True, axis=1)

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
            max_alpha_Q=args.max_alpha_Q,
            min_alpha_Q=args.min_alpha_Q,
            lr_decay_steps_Q=args.lr_decay_steps_Q,
            lr_decay_steps_var=args.lr_decay_steps_var,
            max_alpha_var=args.max_alpha_var,
            min_alpha_var=args.min_alpha_var,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            n_tar_policies=n_target_pol,
            online_alpha=online_alpha,
            exploration_steps=args.exploration_steps,
            temperature_steps=args.temperature_steps,
            type_behavior_policy=type_behavior_policy,
            exp_next_val=exp_next_val,
            num_reward_obj=num_reward_obj
            )

    state_values = defaultdict(list)  # (target_ind, num_states ind)
    true_vals = get_true_val_scaled_exp(mdp, n_target_pol, num_reward_obj, num_states, td.R,
                                        gamma)  # [n pols, n reward objs, s]

    for iter_ in range(num_iter):
        val_target = td.get_V() # nrs
        consumed_step, _ = td.update(behavior_policy, total_env_steps)
        total_env_steps += consumed_step
        # get numerator
        numerator, _ = td.compute_objective(td.get_var(), target_policies, type_behavior_policy)
        denominator = np.linalg.norm(numerator, ord=1, keepdims=True, axis=1)
        for s in range(num_states):
            behavior_policy[s, :] = numerator[s, :] / denominator[s]
        behavior_policy = clip_normalize_policy(behavior_policy, args.beh_min, target_policies)

        if iter_ %400 == 0:
            metrics = {}
            mse_val = np.mean(np.square(true_vals - val_target))
            metrics["mse_avg"] = mse_val
            metrics["num_samples"] = total_env_steps
            wandb.log(metrics)
            print(f"avg mse: {mse_val} at iter :{iter_}")


    wandb.finish()
