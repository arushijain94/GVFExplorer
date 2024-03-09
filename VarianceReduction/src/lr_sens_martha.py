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
from src.common import LinearScheduleClass, get_target_policies, plot_mat_single_dual_goal, get_reward_at_each_step, get_mean_reward
import numpy as np
# import seaborn as sns
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
class TD:
    def __init__(self, mdp, max_steps, num_episodes, S, A, P, size, target_pol, gamma, reward_alpha,
                 n_tar_policies, exploration_steps, temperature_steps, type_behavior_policy, exp_next_val,
                 max_alpha, min_alpha, lr_decay_steps,
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
        self.max_alpha = max_alpha  # learning rate
        self.min_alpha = min_alpha
        self.lr_decay_steps = lr_decay_steps
        self.curr_alpha = max_alpha

        self.reward_alpha = reward_alpha
        self.gamma = gamma
        self.target_pol = target_pol
        self.n_tar_policies = n_tar_policies
        self.online_mean = np.zeros((self.n_tar_policies, self.S, self.A))
        self.online_var = np.ones((self.n_tar_policies, self.S, self.A))
        self.online_alpha = online_alpha
        self.extra_reward_info = 1
        self.var_max = np.ones((self.n_tar_policies, self.S, self.A))*0.01
        self.goal_count_array = np.zeros(4)
        self.type_behavior_policy = type_behavior_policy
        self.exp_next_val = exp_next_val

        self.val_func = np.ones((self.n_tar_policies, self.S))  # V(n, s) func
        self.sr_func = np.zeros((self.n_tar_policies, self.S, self.A, self.S))  # sr = (n,s,a,a)
        self.q_predicted = np.ones((self.n_tar_policies, self.S, self.A))  # sr = (n,s,a,a)

        self.Q_behavior = np.zeros((self.S, self.A))  # sr(s,a,n)
        self.reward_weights = np.zeros((self.n_tar_policies, self.S))

        self.s_a_list = self.get_s_a_pairs()
        self.len_s_a_list = len(self.s_a_list)
        self.epsilon_schedule = LinearScheduleClass(exploration_steps,
                                                    initial_p=1.0,
                                                    final_p=0.)
        self.temp_schedule = LinearScheduleClass(temperature_steps,
                                                    initial_p=1.0,
                                                    final_p=0.1)
        self.lr_schedule = LinearScheduleClass(self.lr_decay_steps,
                                               initial_p=self.max_alpha,
                                               final_p=self.min_alpha)
        self.temp = 1.0

    def get_s_a_pairs(self):
        return list(itertools.product(range(self.S), range(self.A)))

    def set_beh_policy(self, policy):
        self.behavior_pol = policy
        # print("beh:", self.behavior_pol)
        self.update_val_target()

    def update_sr_func(self, s_t, a_t, s_tp1, r_t, combined_R_SF_weight_change=False, intrinsic_reward=0):
        # expected sarsa
        delta= 1. + self.gamma * np.einsum('na,na->n',self.target_pol[:, s_tp1, :], np.mean(self.sr_func, axis=-1)[:,s_tp1,:]) - self.sr_func[:, s_t, a_t, s_tp1]
        self.sr_func[:, s_t, a_t, s_tp1] += self.curr_alpha * delta

        #q update
        delta_q = r_t + self.gamma * np.einsum('na,na->n',self.target_pol[:, s_tp1, :], self.q_predicted[:, s_tp1,:]) - self.q_predicted[:, s_t, a_t]
        self.q_predicted[:,s_t, a_t] += self.curr_alpha * delta_q

        # update the weights of reward function
        reward_delta = r_t - self.reward_weights[:,s_t]
        self.reward_weights[:,s_t] += self.curr_alpha * reward_delta
        # behavior policy update has intrinsic reward as : sum_j |w_{t+1} - w_{t}| of SR
        # behavior_reward = np.sum(np.abs(delta))
        behavior_reward = np.sum(np.abs(self.curr_alpha * delta))
        if combined_R_SF_weight_change:
            behavior_reward += np.sum(np.abs(self.curr_alpha * reward_delta)) + intrinsic_reward
        self.update_behavior_policy_Q(s_t, a_t, s_tp1, behavior_reward)

    def update_behavior_policy_Q(self, s_t, a_t, s_tp1, r):
        # here r is a scalar value
        delta_for_beh = r + self.gamma* np.dot(self.behavior_pol[s_tp1], self.Q_behavior[s_tp1]) - self.Q_behavior[s_t, a_t]
        self.Q_behavior[s_t, a_t] += self.curr_alpha * delta_for_beh


    def get_behavior_policy(self, greed_policy =False, tau=1.):
        policy = np.zeros((self.S, self.A))
        if greed_policy:
            max_action_ind = np.argmax(self.Q_behavior, axis=-1)
            policy[np.arange(self.S), max_action_ind] = 1.
            policy = np.clip(policy, 1e-3, 1.)
            policy = policy/np.linalg.norm(policy, ord=1, keepdims=True, axis=1)
            return policy
        # softmax over the Q value sof behavior, where Q isover Behavior policy intrinsic reward
        max_Q = np.max(self.Q_behavior, axis=1, keepdims=True)
        adjusted_Q = (self.Q_behavior - max_Q)/tau # Adjusted for temperature

        exps_num = np.exp(adjusted_Q)
        exps_denom = np.sum(exps_num, axis=1, keepdims=True)
        pol_dist = exps_num/exps_denom
        pol_dist = np.clip(pol_dist, 0., 1.)
        denom = np.linalg.norm(pol_dist, ord=1, keepdims=True, axis=1)
        pol_dist = pol_dist/denom
        return pol_dist


    def update_sr(self, s_a_s_next_r, combined_R_SF_weight_change):
        for i in range(len(s_a_s_next_r)):
            s, a, s_next, r = s_a_s_next_r[i]
            self.update_sr_func(s, a, s_next, r, combined_R_SF_weight_change=combined_R_SF_weight_change)


    def update_val_target(self):
        self.val_func = np.einsum('nsa,nsa->ns', self.target_pol, self.q_predicted)

    def get_V(self):
        return np.einsum('nsa,nsa->ns', self.target_pol, self.q_predicted)

    def get_val_s(self, state):
        # get Q_pi(n,s,a) = <\sr(n,s,a), w(n,s,a)>
        return self.val_func[:,state]


    def choose_beh_action(self, epsilon, temp, s, pol):
        if np.random.random() <= epsilon:
            a = np.random.choice(self.A)
        else:
            a = np.random.choice(self.A, p=pol[int(s)])
        return a

    def update(self, behavior_policy, total_steps, combined_R_SF_weight_change):
        self.set_beh_policy(behavior_policy)
        step = 0
        s_a_s_next_r_list = []
        epsilon = self.epsilon_schedule.value(total_steps)
        self.temp = self.temp_schedule.value(total_steps)
        self.curr_alpha = self.lr_schedule.value(total_steps)
        for e in range(self.num_episodes):
            obs = self.env.reset()
            s = np.argmax(obs)
            done = False
            env_steps = 0
            while not done and env_steps < self.max_steps:
                a = self.choose_beh_action(epsilon, self.temp, s, self.behavior_pol)
                obs_next, _, done, info = self.env.step(a)
                r= get_reward_at_each_step(self.env_name, self.n_tar_policies, self.S, self.A, self.size, s, a, std_dev_reward=5)
                s_next = np.argmax(obs_next)
                step += 1
                s_a_s_next_r = (int(s), int(a), int(s_next), r)
                s_a_s_next_r_list.append(s_a_s_next_r)
                s = s_next
                env_steps += 1
        self.update_sr(s_a_s_next_r_list, combined_R_SF_weight_change)
        return step

    def get_val(self):
        return self.val_func

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
        # print(f"Steps: {num_steps} and state visit {state_visitation}")
        return plot_mat_single_dual_goal(self.env_name, state_visitation, self.size, num_steps, 'Visitation Frequency')


    def plot_value(self, num_steps, log_scale=False):
        val_mat = self.get_val()[:,:-1] # [n s]
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

    def plot_sr(self, num_steps, log_scale=False):
        #sr = [n,sat] -> ns
        val_mat = np.mean(self.sr_func,axis=(2,3))[:,:-1] # [n s]
        val_mat = val_mat.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            val_mat = np.log(val_mat + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, val_mat[i], self.size, num_steps, f"sr_{i}"+x)
            wandb.log({"num_samples": num_steps,
                       f"sr_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt = plot_mat_single_dual_goal(self.env_name, np.mean(val_mat, axis=0), self.size, num_steps, "MeanSR"+x)
        wandb.log({"num_samples": num_steps,
                   f"MeanSR"+x: wandb.Image(plt)})
        plt.close()

    def plot_reward_weight(self, num_steps, log_scale=False):
        # r = ns
        val_mat = self.reward_weights[:,:-1] # [n s]
        val_mat = val_mat.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            val_mat = np.log(val_mat + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, val_mat[i], self.size, num_steps, f"R_w_{i}"+x)
            wandb.log({"num_samples": num_steps,
                       f"R_w_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt = plot_mat_single_dual_goal(self.env_name, np.mean(val_mat, axis=0), self.size, num_steps, "MeanR_w"+x)
        wandb.log({"num_samples": num_steps,
                   f"MeanR_w"+x: wandb.Image(plt)})
        plt.close()

    def plot_value_diff(self, num_steps, true_val, log_scale=False):
        true_val = true_val[:, :-1]
        var_mat = self.get_val()[:, :-1]  # [n s]
        abs_diff = np.abs(true_val - var_mat)
        abs_diff = abs_diff.reshape((self.n_tar_policies, self.size, self.size))
        x = ""
        if log_scale:
            abs_diff = np.log(abs_diff + 1)
            x="_log"
        for i in range(self.n_tar_policies):
            plt = plot_mat_single_dual_goal(self.env_name, abs_diff[i], self.size, num_steps, f"diff_value_{i}"+x)
            wandb.log({"num_samples": num_steps,
                       f"diff_value_{i}"+x: wandb.Image(plt)})
            plt.close()
        plt = plot_mat_single_dual_goal(self.env_name, np.mean(abs_diff, axis=0), self.size, num_steps, "Mean_diff_value"+x)
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

        weights_num = 1. / num_policies * np.ones_like(var)
        var_num = var + tolerance

        if type_behavior_policy == 1:  # softmax directly on var value
            numerator = softmax_numerator(np.mean(var_num, axis=0), temp)
        elif type_behavior_policy == 2:  # first one but with tolerance in target policy
            new_target_policies = np.clip(target_policies, 0.05, 1.0)
            new_target_policies = new_target_policies / np.linalg.norm(new_target_policies, ord=1, keepdims=True,
                                                                       axis=-1)
            numerator = np.sqrt(np.einsum('nsa,nsa,nsa->sa', np.square(new_target_policies), weights_num, var_num))
        else:
            # type_behavior_policy == 0:  # \mu(s,a) = \sqrt(\sum_i \pi^2(s,a) * var(s,a))
            numerator = np.sqrt(np.einsum('nsa,nsa,nsa->sa', np.square(target_policies), weights_num, var_num))
        return numerator, weights_num

def get_true_vals(mdp, n_target_pol, S, target_policies, R, gamma):
    # get true V_pi
    true_vals = np.ones((n_target_pol, S))
    for i in range(n_target_pol):
        true_vals[i] = np.array(mdp.get_v(R[i], gamma, target_policies[i]))
    return true_vals

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



if __name__ == '__main__':
    '''
    TODO:
    1. Add min clip for action from a policy
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes for tuning PI")
    parser.add_argument("--num_iter", default=100, type=int,
                        help="Total number of iterations of updating the behaviors policy.")
    parser.add_argument("--max_alpha", default=1.0, type=float, help="Learning Rate")
    parser.add_argument("--min_alpha", default=0.5, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay_steps", default=500000, type=int, help="Learning Rate decay in these many steps")

    parser.add_argument("--reward_alpha", default=0.25, type=float, help="Learning Rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="Grid_martha_try1", type=str, help="Group name for Wandb")
    parser.add_argument("--wandb_name", default="1", type=str, help="Log name for Wandb")
    parser.add_argument("--timeout", default=500, type=int, help="Max episode length of the environment.")
    parser.add_argument("--beh_min", default=1e-3, type=float, help="Minimum value of behavior policy allowed")
    parser.add_argument("--logdir", default="/home/mila/j/jainarus/scratch/VarReduction/Det_10/Feb_20/try", type=str, help="Directory for logging")
    parser.add_argument("--size", default=20, type=int, help="size of MDP")
    parser.add_argument("--stochastic_env", default=True, type=bool, help="Type of transition F: det; T:stochastic")
    parser.add_argument("--n_target_pol", default=2, type=int, help="Num of Target Policies")
    parser.add_argument("--random_start_beh", default=2, type=int, help="(0: one of target policy as start policy, 1: random policy, 2: mixture target policies)")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--online_alpha", default=0.1, type=float, help="alpha for normalization")
    parser.add_argument("--env_name", default="online_rooms_same_reward_grid", type=str, help="{simple_grid, online_rooms_same_reward_grid, online_rooms_different_reward_grid}")
    parser.add_argument("--type_obj", default=1, type=int, help="{1:var_sum, 2:norm_var_sum, 3:var_weighted_sum, 4:norm_var_weighted_sum}")
    parser.add_argument("--last_k", default=500, type=int, help="last k values for moving reward func")
    parser.add_argument("--exploration_steps", default=500, type=int, help="exploration steps")
    parser.add_argument("--type_behavior_policy", default=0, type=int, help="(0: softmax, 1: eps-reedy)")
    parser.add_argument("--temperature_steps", default=200, type=int, help="temperature steps")
    parser.add_argument("--exp_next_val", default=1, type=int, help="(0: use rho 1: exp under pi)")
    parser.add_argument("--behv_reward_combined", default=1, type=int, help="(0: only weight change in SR,  1: SR+reward weight change)")


    args = parser.parse_args()
    np.random.seed(args.seed)
    last_k = args.last_k

    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_dir = f"{args.wandb_group}_{current_time}_{args.min_alpha}_{args.lr_decay_steps}"
    wandb_dir = f"/home/mila/j/jainarus/scratch/VarReduction/wandb/{wandb_dir}"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    # initializing wandb for storing results
    wandb.init(project="simple-gvf-evaluations", entity="jainarus", group=args.wandb_group,
               config=namespace_to_dict(args),
               name="seed_"+ str(args.seed),
               dir=wandb_dir,
               mode="online")

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

    target_policies = get_target_policies(num_states, args.env_name, args.n_target_pol)

    min_alpha = args.min_alpha
    lr_q_list = [0.1, 0.25, 0.5, 0.8, 0.9, 0.95]
    mse_value_list = []
    samples_consumed = []

    for i in range(len(lr_q_list)):
        min_alpha = lr_q_list[i]
        metrics = {}

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
        args.reward_alpha = min_alpha
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
                reward_alpha=args.reward_alpha,
                n_tar_policies=n_target_pol,
                online_alpha=online_alpha,
                exploration_steps=args.exploration_steps,
                temperature_steps=args.temperature_steps,
                type_behavior_policy=type_behavior_policy,
                exp_next_val=exp_next_val,
                )
        state_values = defaultdict(list)  # (target_ind, num_states ind)
        true_vals = get_true_vals(mdp, n_target_pol, num_states, target_policies, get_mean_reward(args.env_name, n_target_pol, num_states, num_actions, args.size), gamma)


        for iter_ in range(num_iter):
            total_env_steps += td.update(behavior_policy, total_env_steps, args.behv_reward_combined)
            behavior_policy = td.get_behavior_policy(greed_policy=type_behavior_policy)
            val_target = td.get_V()
            for tar_ind in range(n_target_pol):
                for s in range(num_states):
                    state_values[(tar_ind, s)].append(val_target[tar_ind, s])

            if total_env_steps>500000:
                # get value estimate from the learnt value function
                val_funcs = np.zeros((n_target_pol, num_states))
                mse=np.zeros((n_target_pol, num_states))
                for tar_ind in range(n_target_pol):
                    for s in range(num_states):
                        estimated_values = state_values[(tar_ind, s)]
                        estimated_values = estimated_values[-last_k:] if len(estimated_values)>=last_k else estimated_values
                        mse[tar_ind,s] = np.mean(np.square(true_vals[tar_ind,s] - estimated_values))

                # n dim metrics for gridworld
                mse_val = np.mean(mse)
                mse_value_list.append(mse_val)
                samples_consumed.append(total_env_steps)
                metrics["LR_Q"] = min_alpha
                metrics["mse_avg"] = mse_val
                metrics["num_samples"] = total_env_steps
                wandb.log(metrics)
                print(f" here ----- alpha: {min_alpha} and avg_mse: {mse_val}")
                break

    for i in range(len(lr_q_list)):
        print(f"alpha: {lr_q_list[i]} and avg_mse: {mse_value_list[i]} at samples: {samples_consumed[i]}")
    wandb.finish()

