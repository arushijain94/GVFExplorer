### Q:
# 1. How to handle zeros in value functions
# 2. Should we log environment steps?
# 3. Should we also log the density of state visitation distribution?
import sys
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from src.common import *
from src.common import LinearScheduleClass, get_target_policies, plot_mat_single_dual_goal, get_reward_at_each_step, get_mean_reward
import numpy as np
import wandb
import itertools
from collections import defaultdict
import src.common as common
from src.common import *
from src.common import LinearScheduleClass, get_target_policies, plot_mat_single_dual_goal, get_reward_at_each_step, get_mean_reward
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

class TD:
    def __init__(self, mdp, max_steps, num_episodes, S, A, P, size, target_pol, gamma, max_alpha, min_alpha, lr_decay_steps,
                 rho_min, rho_max,
                 n_tar_policies, exploration_steps, ):
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
        self.lr_decay_steps = lr_decay_steps
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.target_pol = target_pol
        self.n_tar_policies = n_tar_policies
        self.val_func = np.ones((self.n_tar_policies, self.S, self.A))  # Q(s,a) func
        self.extra_reward_info = 1
        self.s_a_list = self.get_s_a_pairs()
        self.epsilon_schedule = LinearScheduleClass(exploration_steps,
                                                    initial_p=1.0,
                                                    final_p=0.)
        self.lr_schedule = LinearScheduleClass(self.lr_decay_steps,
                                               initial_p=self.max_alpha,
                                               final_p=self.min_alpha)  # linearly decay lr with every step


    def get_s_a_pairs(self):
        return list(itertools.product(range(self.S), range(self.A)))

    def choose_beh_action(self, epsilon, s, pol):
        if np.random.random() <= epsilon:
            a = np.random.choice(self.A)
        else:
            a = np.random.choice(self.A, p=pol[int(s)])
        return a

    def set_beh_policy(self, policy):
        self.behavior_pol = policy
        self.rho = np.array([set_rho(self.target_pol[i], self.behavior_pol, self.rho_min, self.rho_max) for i in range(self.n_tar_policies)])

    def get_val_s(self, state):
        return np.einsum('na,na->n', self.target_pol[:, state, :], self.val_func[:, state, :])

    def update_val_func(self, s_t, a_t, s_tp1, r_t):
        # expected sarsa
        delta = r_t + self.gamma * self.get_val_s(s_tp1) - self.val_func[:, s_t, a_t]
        self.val_func[:, s_t, a_t] += self.curr_alpha * delta
        # print(f" val:{self.val_func[0]}")

    def get_V(self):
        return np.einsum('nsa,nsa->ns', self.target_pol, self.val_func)

    def update_val(self, s_a_s_next_r):
        for i in range(len(s_a_s_next_r)):
            s, a, s_next, r = s_a_s_next_r[i]
            self.update_val_func(s, a, s_next, r)

    def update(self, behavior_policy, theta, total_steps):
        num_steps = 0
        self.set_beh_policy(behavior_policy)
        update_theta = np.zeros_like(theta)
        epsilon = self.epsilon_schedule.value(total_steps)
        self.curr_alpha = self.lr_schedule.value(total_steps)
        s_a_s_next_r_list = []
        for e in range(self.num_episodes):
            obs = env.reset()
            s = np.argmax(obs)
            done = False
            env_steps = 0
            g = np.zeros(self.n_tar_policies)
            rho_ = np.ones(self.n_tar_policies)
            gamma_ = 1.0
            visited = np.zeros((num_states, num_actions))
            while not done and env_steps < self.max_steps:
                env_steps += 1
                a = self.choose_beh_action(epsilon, s, self.behavior_pol)
                obs_next, _, done, info = self.env.step(a)
                r = get_reward_at_each_step(self.env_name, self.n_tar_policies, self.S, self.A, self.size, s, a,
                                            std_dev_reward=5)

                s_next = np.argmax(obs_next)
                rho_ *= self.rho[:,s,a]
                g += gamma_ * r
                s_a_s_next_r = (int(s), int(a), int(s_next), r)
                s_a_s_next_r_list.append(s_a_s_next_r)
                gamma_ *= gamma
                visited[s,a] += 1
                s = s_next

            num_steps +=env_steps

            update_theta += np.sum(np.square(g*rho_)) * visited * (1. - self.behavior_pol)
            # print(f"g: {g}, rho:{rho_}, update theta: {update_theta}")

        self.update_val(s_a_s_next_r_list)
        update_theta /= num_episodes
        # update behavior policy
        theta = theta + self.curr_alpha * update_theta # update theta
        theta += 1e-6

        return theta, num_steps


def get_true_vals(mdp, n_target_pol, S, target_policies, R, gamma):
    # get true V_pi
    true_vals = np.ones((n_target_pol, S))
    for i in range(n_target_pol):
        true_vals[i] = np.array(mdp.get_v(R[i], gamma, target_policies[i]))
    return true_vals


if __name__ == '__main__':
    '''
    TODO:
    1. Add min clip for action from a policy
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes for tuning PI")
    parser.add_argument("--num_iter", default=50, type=int,
                        help="Total number of iterations of updating the behaviors policy.")
    parser.add_argument("--max_alpha", default=1.0, type=float, help="Learning Rate")
    parser.add_argument("--min_alpha", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay_steps", default=500000, type=int, help="Learning Rate decay in these many steps")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--wandb_group", default="bpg", type=str, help="Group name for Wandb")
    parser.add_argument("--timeout", default=300, type=int, help="Max episode length of the environment.")
    parser.add_argument("--rho_max", default=1, type=float, help="Maximum value of Rho allowed.")
    parser.add_argument("--rho_min", default=1e-3, type=float, help="Minimum value of Rho allowed")
    parser.add_argument("--size", default=20, type=int, help="size of MDP")
    parser.add_argument("--stochastic_env", default=True, type=bool, help="Type of transition F: det; T:stochastic")
    parser.add_argument("--n_target_pol", default=2, type=int, help="Num of Target Policies")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--logdir", default="/home/mila/j/jainarus/scratch/VarReduction/Det_10/Feb_20/try", type=str, help="Directory for logging")
    parser.add_argument("--env_name", default="online_rooms_same_reward_grid", type=str,
                        help="{simple_grid, online_rooms_same_reward_grid, online_rooms_different_reward_grid,"
                             "semigreedy_online_rooms_same_reward_grid, semigreedy_online_rooms_different_reward_grid}")
    parser.add_argument("--last_k", default=500, type=int, help="last k values for moving reward func")
    parser.add_argument("--exploration_steps", default=200, type=int, help="exploration steps")

    args = parser.parse_args()
    np.random.seed(args.seed)
    last_k = args.last_k

    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_dir = f"{args.wandb_group}_{current_time}_{args.min_alpha}_{args.seed}"
    wandb_dir = f"/home/mila/j/jainarus/scratch/VarReduction/wandb/{wandb_dir}"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    # common.set_seed(args.seed) # seed for common program

    # initializing wandb for storing results
    wandb.init(project="simple-gvf-evaluations", entity="jainarus", group=args.wandb_group,
               config=namespace_to_dict(args),
               name="seed_" + str(args.seed),
               dir=wandb_dir)

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # deterministic 5X5 MDP
    mdp = common.create_env(args.env_name, size=args.size, gamma=args.gamma, stochastic_transition=args.stochastic_env)
    num_states = mdp.S
    num_actions = mdp.A
    gamma = mdp.gamma
    env = mdp.env

    num_iter = args.num_iter
    num_episodes = args.num_episodes
    n_target_pol = args.n_target_pol

    target_policies = get_target_policies(num_states, args.env_name, args.n_target_pol)


    #### Behavior
    behavior_policy = np.mean(target_policies, axis=0)
    behavior_policy = behavior_policy / np.linalg.norm(behavior_policy, ord=1, keepdims=True, axis=1)
    theta = np.copy(behavior_policy)

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
            min_alpha = args.min_alpha,
            lr_decay_steps=args.lr_decay_steps,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            n_tar_policies=n_target_pol,
            exploration_steps=args.exploration_steps,
            )
    state_values = defaultdict(list)  # (target_ind, num_states ind)
    true_vals = get_true_vals(mdp, n_target_pol, num_states, target_policies,
                              get_mean_reward(args.env_name, n_target_pol, num_states, num_actions, args.size), gamma)

    for iter_ in range(num_iter):
        theta, num_steps = td.update(behavior_policy, theta, total_env_steps)
        # print(f" iter: {iter_}, theta:{theta}")
        behavior_policy = theta / np.linalg.norm(theta, ord=1, keepdims=True, axis=1)
        total_env_steps += num_steps
        val_target = td.get_V()
        for tar_ind in range(n_target_pol):
            for s in range(num_states):
                state_values[(tar_ind, s)].append(val_target[tar_ind, s])

        if iter_ %500 ==0:
            mse = np.zeros((n_target_pol, num_states))
            for tar_ind in range(n_target_pol):
                for s in range(num_states):
                    estimated_values = state_values[(tar_ind, s)]
                    estimated_values = estimated_values[-last_k:] if len(
                        estimated_values) >= last_k else estimated_values
                    mse[tar_ind, s] = np.mean(np.square(true_vals[tar_ind, s] - estimated_values))

            mse_metrics = np.mean(mse, axis=1)

            metrics = {}
            for i in range(n_target_pol):
                metrics["mse_" + str(i + 1)] = mse_metrics[i]

            metrics["mse_avg"] = np.mean(mse_metrics)
            metrics["num_samples"] = total_env_steps

            # wandb logging
            wandb.log(metrics)

    wandb.finish()
    np.save(os.path.join(args.logdir, "final_beh_" + str(total_env_steps) + ".npy"), behavior_policy)

