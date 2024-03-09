import argparse
import envs
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.common import MultiRewardReplayBuffer, MultiRewardNextActionReplayBuffer, PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from src.common import load_config, linear_schedule, make_env, load_state_to_val_dict, save_temp_config, create_env_args,  LinearScheduleClass, discretize_state
import wandb
import warnings
from src.multi_dim_dqn import DQNAgent
from src.simple_target_policy import SimpleTargetPolicy
import copy
import yaml
import sys
import io
from PIL import Image
from math import sqrt
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", default=None)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="the group (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--run-save-loc", type=str, default="",
                        help="save location for the data")
    parser.add_argument("--q-type-loss", type=int, default=0,
                        help="[0: l1 loss, 1: mse loss]")
    parser.add_argument("--var-type-loss", type=int, default=0,
                        help="[0: l1 loss, 1: mse loss]")



    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--goal-position", type=float, default=0.0, help="position of goal for pendulum np.pi") # goal position for target_policies, don't modify the yml for this!
    parser.add_argument("--goal-positions-pe", type=float, nargs=2, default=[0.0, -0.78],
                        help="Two positions of goal for pendulum, separated by a space") # goal position for PE

    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--start-var-lambd", type=float, default=1.,
                        help="the starting lambda for exploration and var")
    parser.add_argument("--end-var-lambd", type=float, default=0.,
                        help="the ending lambda for exploration and var")
    parser.add_argument("--var-lambd-fraction", type=float, default=0.001,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--variance-train-frequency", type=int, default=2,
                        help="the frequency of training var network")
    parser.add_argument("--type-next-target-val", type=int, default=0,
                        help="{0: Q(s,a) = r(s,a) + y E_pi[pi(s',a') Q(s,', a')],  1:Q(s,a) = r(s,a) + y rho(s',a') Q(s,', a')}")

    parser.add_argument("--eval-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--checkpoint-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="the episodes of evaluation")
    parser.add_argument("--beh-plot-frequency", type=int, default=1000,
                        help="frequency of saving beh policy trajectory")

    parser.add_argument("--prioritized-replay", type=int, default=0,
                        help="PER: {0: false, 1: true}")
    parser.add_argument("--prioritized-replay-alpha", type=float, default=0.6,
                        help="prioritized-replay-alpha")
    parser.add_argument("--prioritized-replay-beta", type=float, default=0.4,
                        help="prioritized-replay-alpha")
    parser.add_argument("--prioritized-replay-eps", type=float, default=1e-6,
                        help="prioritized-replay-eps")

    parser.add_argument("--compute-mse-for-random-states", type=int, default=1,
                        help="{0: pi state dist, 1: random states}")

    parser.add_argument(
        "--evaluate-checkpoint", action="store_true", default=False, help="to evaluate a particular model"
    )
    parser.add_argument(
        "--checkpoint-number", type=int, default=100, help="to evaluate a checkpoint"
    )

    # First parse the config file argument
    config_args, remaining_argv = parser.parse_known_args()
    #
    # print(f"config args {config_args}, rema: {remaining_argv}")

    # Load the configuration file if specified
    if config_args.config:
        config = load_config(config_args.config)
        arg_defaults = vars(config_args)
        arg_defaults.update(config)
        parser.set_defaults(**arg_defaults)

    # print("remaining args: ", remaining_argv)

    # Parse remaining command line arguments
    args = parser.parse_args(remaining_argv)
    print("args:", args)
    # print(f"lr: {args.learning_rate}, seed {args.seed}, batch size {args.batch_size}")

    return args


class MultiDimQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, reward_dim):
        super(MultiDimQNetwork, self).__init__()
        self.num_actions = num_actions
        self.reward_dim = reward_dim

        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Separate heads for each reward dim
        self.heads = nn.ModuleList([nn.Linear(64, num_actions) for _ in range(reward_dim)])

    def forward(self, x):
        # outputs a q_val of dim [batch size * action_dim * reward_dim]
        shared_features = self.shared_layers(x)
        q_values = [head(shared_features) for head in self.heads]
        q_vals = torch.stack(q_values, dim=-1)
        return q_vals.reshape((-1, self.num_actions, self.reward_dim))

class ParametricSoftplus(nn.Module):
    def __init__(self):
        super(ParametricSoftplus, self).__init__()
        # Initialize beta as a learnable parameter
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Apply the parametric softplus function
        return F.softplus(self.beta * x)


class MultiDimVarNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, reward_dim):
        super(MultiDimVarNetwork, self).__init__()
        self.num_actions = num_actions
        self.reward_dim = reward_dim
        # self.parametric_softplus = ParametricSoftplus()
        self.softplus_func = nn.Softplus()

        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Separate heads for each reward dim
        self.heads = nn.ModuleList([nn.Linear(64, num_actions) for _ in range(reward_dim)])

    def forward(self, x):
        # outputs a q_val of dim [batch size * action_dim * reward_dim]
        shared_features = self.shared_layers(x)
        var_values = [self.softplus_func(head(shared_features)) for head in self.heads]
        var_vals = torch.stack(var_values, dim=-1)
        return var_vals.reshape((-1, self.num_actions, self.reward_dim))


class Dueling_Q_Var_Net(nn.Module):
    """
    Shared network for Q and Var, such that features are shared
    """
    def __init__(self, state_dim, num_actions, reward_dim, hidden_dim=64):
        super(Dueling_Q_Var_Net, self).__init__()
        self.num_actions = num_actions
        self.reward_dim = reward_dim

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.Q = nn.Linear(hidden_dim, self.num_actions * self.reward_dim)
        self.Var = nn.Linear(hidden_dim, self.num_actions * self.reward_dim)

    def forward(self, x, update_q_only=False, update_var_only=False):
        shared_features = self.shared_layers(x)
        Q = self.Q(shared_features)
        Var = self.Var(shared_features)

        # Apply Softplus to ensure Var outputs are non-negative
        Var = F.softplus(Var)
        # Reshape A to match the dimensions of V
        Q = Q.view(-1, self.num_actions, self.reward_dim) # batch_size X action_dim X reward_dim
        Var = Var.view(-1, self.num_actions, self.reward_dim)  # batch_size X action_dim X reward_dim

        if update_q_only:
            Var = Var.detach()
            return Q
        elif update_var_only:
            Q = Q.detach()
            return Var


class VariancePolicyEvaluationAgent:
    # target_policy_agents: list of target policy agents.

    def __init__(self, args, target_policy_agents):
        self.args =args
        self.num_target_policies = len(target_policy_agents)
        self.target_policy_agents = target_policy_agents # target policy agent networks list
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.infty_approx = 1e5
        self.var_lambda = 0.

        name_goals = f"goal_tp_{self.args.puddle_goal_type}"
        run_name = f"{args.env_id}__{args.exp_name}__{name_goals}__LR{self.args.learning_rate}__Buffer{self.args.buffer_size}__VarLambd{self.args.var_lambd_fraction}__VarFreq{self.args.variance_train_frequency}__tau{self.args.tau}__Seed{self.args.seed}"
        self.args.run_name = run_name

        self.args.run_save_loc = os.path.join(self.args.run_save_loc, run_name)
        print("new locations in PE: *******", self.args.run_save_loc)
        if not os.path.exists(self.args.run_save_loc):
            print("run save path exist:", self.args.run_save_loc)
            os.makedirs(self.args.run_save_loc)

        if args.track:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                group=self.args.wandb_group,
                sync_tensorboard=True,
                config=vars(self.args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        self.writer = SummaryWriter(f"{self.args.run_save_loc}/runs")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        # env setup
        env_args = create_env_args(args, args.env_id)
        # print("eval arguments of env:", env_args)
        self.env = make_env(self.args.env_id, self.args.seed, self.args.capture_video, run_name, env_args) # training
        self.eval_env = make_env(self.args.env_id, self.args.seed, self.args.capture_video, run_name, env_args)

        # print("env:", self.env)

        self.observation_space = self.env.observation_space
        self.state_dim = np.array(self.observation_space.shape).prod()
        self.action_space = self.env.action_space
        self.action_dim = self.action_space.n
        self.reward_dim = self.env.reward_space.n

        # print(f"obse: {self.observation_space}, action {self.action_space}, reward {self.reward_dim}")

        assert isinstance(self.action_space, gym.spaces.Discrete), "only discrete action space is supported"

        # duelling network
        self.duel_network = Dueling_Q_Var_Net(self.state_dim, self.action_dim, self.reward_dim).to(self.device)
        self.target_duel_network = Dueling_Q_Var_Net(self.state_dim, self.action_dim, self.reward_dim).to(self.device)
        self.optimizer = optim.Adam(self.duel_network.parameters(), lr=self.args.learning_rate)
        self.target_duel_network.load_state_dict(self.duel_network.state_dict())
        self.target_duel_network.eval()

        if self.args.prioritized_replay:
            self.rb = PrioritizedReplayBuffer(
                self.args.buffer_size,
                self.args.prioritized_replay_alpha,
                self.observation_space,
                self.action_space,
                self.device,
                handle_timeout_termination=True,
                reward_dim=self.reward_dim,
            )
            self.beta_schedule = LinearScheduleClass(int(8000),
                                       initial_p=self.args.prioritized_replay_beta,
                                       final_p=1.0)
        else:
            # simple replay buffer
            self.rb = MultiRewardNextActionReplayBuffer(
                self.args.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                handle_timeout_termination=True,
                reward_dim=self.reward_dim,
            )
        self.epsilon_schedule = LinearScheduleClass(int(5000),
                                       initial_p=self.args.start_e,
                                       final_p=self.args.end_e)



        # reset env
        self.obs, _ = self.env.reset() #seed=self.args.seed
        # self.action = self.select_behavior_action(self.obs
        self.train_episodic_return = np.zeros(self.reward_dim)
        self.train_episodic_len = 0
        self.action= None

        # var lambda schedule
        self.var_lambda_schedule = LinearScheduleClass(int(8000),
                                                    initial_p=self.args.start_var_lambd,
                                                    final_p=self.args.end_var_lambd)

        # save config
        save_temp_config(self.args.run_save_loc, self.args)

    # Function to copy network weights
    def get_weights_copy(self, net):
        return {name: param.clone() for name, param in net.named_parameters()}

    def select_behavior_action(self, obs, epsilon, var_lambda):

        # for single observation
        if random.random() < epsilon:
            actions = np.array([self.env.action_space.sample()])
        else:
            with torch.no_grad():
                behavior_policy_prob = self.get_behavior_policy_prob(obs, self.target_policy_agents, var_lambda) # [na] dim
                # behavior_policy_prob = behavior_policy_prob / behavior_policy_prob.sum(dim=1, keepdim=True)
                # Check that they now sum to 1
                if not torch.allclose(behavior_policy_prob.sum(dim=1), torch.tensor(1.0)):
                    raise ValueError("Behavioir Probabilities do not sum to 1.")
                actions = torch.multinomial(behavior_policy_prob, num_samples=1).reshape(-1).cpu().numpy()
        return actions

    def select_action(self, obs, epsilon):
        # for single observation
        if random.random() < epsilon:
            actions = np.array([self.env.action_space.sample()])
        else:
            with torch.no_grad():
                # behavior_policy \prop \sum_i \pi_i(a|s) \sqrt{var_network_i(s,a)}
                current_target_policy_agent = self.target_policy_agents[
                    self.current_target_policy_index]  # select target policy net
                action_probs = self.get_target_policy_prob(obs, current_target_policy_agent)
                actions = torch.multinomial(action_probs, num_samples=1).reshape(-1).cpu().numpy()
        return actions

    def collect_rollout(self, global_step):
        # 1 step execution of env
        epsilon = self.epsilon_schedule.value(global_step)

        # annealing of lambda parameter controlling exploration and variance values
        self.var_lambda = self.var_lambda_schedule.value(global_step)

        # print(f" collect rollout: lambd {var_lambda}, gloabl step {global_step}, start {self.args.start_var_lambd}, end : {self.args.end_var_lambd}, dur {self.args.var_lambd_fraction * self.args.total_timesteps}")

        if self.action is None:
            self.action = self.select_behavior_action(self.obs, epsilon, self.var_lambda) # select action from behavior policy agent
        # print(f"actions: {actions}, ad shape: {actions.shape}, type: {type(actions)}")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = self.env.step(self.action[0])

        self.train_episodic_return += rewards
        self.train_episodic_len +=1
        # print(f"obs {next_obs}, reward: {rewards}, term {terminations} trunc {truncations} info {infos}")
        dones = terminations or truncations
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        if dones is True:
            next_actions = np.zeros_like(self.action)
        else:
            next_actions = self.select_behavior_action(real_next_obs, epsilon, self.var_lambda)
        real_next_action = next_actions.copy()

        self.rb.add(self.obs, real_next_obs, self.action, real_next_action, rewards, dones, infos)

        if dones:
            for i, ep_reward in enumerate(self.train_episodic_return):
                # Log each dimension separately
                self.writer.add_scalar(f"train/episodic_return_{i+1}", ep_reward, global_step)
            self.writer.add_scalar(f"train/episodic_return_avg", np.mean(self.train_episodic_return), global_step)
            self.writer.add_scalar("train/episodic_length", self.train_episodic_len, global_step)
            self.writer.add_scalar("train/epsilon", epsilon, global_step)
            self.writer.add_scalar("train/var_lambda", self.var_lambda, global_step)
            # self.writer.add_scalar("train/beta_for_softmax", self.var_network.parametric_softplus.beta.item(), global_step)
            # reset env
            self.obs, _ = self.env.reset() #seed=self.args.seed
            self.action = None
            self.train_episodic_return = np.zeros(self.reward_dim)
            self.train_episodic_len = 0
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            self.obs = next_obs
            self.action = next_actions

    def get_discretized_value_mat(self, net, q=False, var=False, num_samples=5, bins=20):
        mat_counts = np.zeros((2, bins, bins))
        bin_size = 1./bins
        for i in range(0, bins):
            for j in range(0, bins):
                start_x = i*bin_size
                start_y = j*bin_size
                data = np.random.uniform((start_x,start_y), (start_x+bin_size, start_y+bin_size), (num_samples, 2))
                data = np.clip(data, a_min=0., a_max=1.)
                with torch.no_grad():
                    if q==True:
                        val = net(torch.Tensor(data).to(self.device), update_q_only=True).mean(dim=(0,1)) #r
                    elif var==True:
                        val = net(torch.Tensor(data).to(self.device), update_var_only=True).mean(dim=(0, 1))  # r
                    mat_counts[:, i, j] = val.cpu().detach().numpy()
        # print("mat count: ", mat_counts)
        return mat_counts

    def plot_matrix(self, mat, name, global_step):
        plt, fig = self.eval_env.plot_mat(mat)
        # Instead of directly logging the figure to WandB, save it to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Load the image from the buffer into PIL
        image = Image.open(buf)

        # Convert the image to RGB (if it's not already in RGB)
        image = image.convert('RGB')

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert the numpy array to PyTorch tensor
        # Permute the dimensions to [C, H, W] as expected by TensorBoard
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        # Use the SummaryWriter to add the figure
        self.writer.add_image(f'{name}/step_{global_step}', image_tensor, global_step)
        buf.close()

    def plot_variance_func(self, global_step):
        var_mat = self.get_discretized_value_mat(self.duel_network, var=True)
        self.plot_matrix(var_mat[0], "Variance1", global_step)
        self.plot_matrix(var_mat[1], "Variance2", global_step)
        self.plot_matrix(np.mean(var_mat, axis=0), "MeanVariance", global_step)
        print("plotted variance")

    def plot_value_func(self, global_step):
        q_mat = self.get_discretized_value_mat(self.duel_network, q=True)
        self.plot_matrix(q_mat[0], "Q1", global_step)
        self.plot_matrix(q_mat[1], "Q2", global_step)
        self.plot_matrix(np.mean(q_mat, axis=0), "MeanQ", global_step)
        print("plotted q")

    def get_trajectories_for_state_visitation_beh_policy(self, global_step, bins=20, num_trajectories=5000):
        # print("in trajectory plotting")
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                              zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]

        visitation_counts = np.zeros((bins, bins))
        for traj in range(num_trajectories):
            obs, _ = self.eval_env.reset()
            done = False
            max_steps = 200
            curr_step = 0
            while not done and curr_step <= max_steps:
                actions = self.select_behavior_action(obs, epsilon=0, var_lambda=0)
                next_obs, rewards, terminations, truncations, _ = self.eval_env.step(actions[0])
                # Discretize the state and update visitation counts
                discretized_state = discretize_state(obs, bins_per_dimension)
                visitation_counts[discretized_state[0]-1][discretized_state[1]-1] += 1
                done = terminations or truncations
                obs = next_obs
                curr_step += 1
            # print(f"traj done {traj}")
        # print(f"done get traj")
        plt, fig = self.eval_env.plot_mat(visitation_counts)
        # Instead of directly logging the figure to WandB, save it to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Load the image from the buffer into PIL
        image = Image.open(buf)

        # Convert the image to RGB (if it's not already in RGB)
        image = image.convert('RGB')

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert the numpy array to PyTorch tensor
        # Permute the dimensions to [C, H, W] as expected by TensorBoard
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        # Use the SummaryWriter to add the figure
        self.writer.add_image(f'visitation_{global_step}/state_visitation', image_tensor, global_step)
        print("added in wandb now")
        buf.close()

    def discretize_state(self, state, bins_per_dimension):
        discrete_state = tuple(np.digitize(s, b) for s, b in zip(state, bins_per_dimension))
        return discrete_state

    def get_discretized_value_abs_diff(self, net, state_to_true_val_dict_list, bins=20):
        mat_counts = np.zeros((2, bins, bins))
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                                   zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]

        for i in range(self.reward_dim):
            state_to_val = state_to_true_val_dict_list[i]
            states, true_values = zip(*state_to_val.items())
            state_tensor = torch.tensor(states)
            true_value_tensor = torch.tensor(true_values)
            with torch.no_grad():
                estimated_s_a_vals = net(state_tensor, update_q_only=True).to(self.device)
                estimated_v_vals = self.get_exp_v_target(estimated_s_a_vals, state_tensor)[:, i]
                estimated_v_vals = estimated_v_vals.reshape(-1)
                abs_diff_error = torch.abs(estimated_v_vals - true_value_tensor)
            abs_diff_mat = abs_diff_error.cpu().detach().numpy()
            for ind, s in enumerate(states):
                disc_state = self.discretize_state(s, bins_per_dimension)
                mat_counts[i, int(disc_state[0] - 1),int(disc_state[1] - 1)] = abs_diff_mat[ind]
        return mat_counts

    def get_discretized_true_q(self, state_to_true_val_dict_list, bins=20):
        mat_counts = np.zeros((2, bins, bins))
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                                   zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]

        for i in range(self.reward_dim):
            state_to_val = state_to_true_val_dict_list[i]
            states, true_values = zip(*state_to_val.items())
            state_tensor = torch.tensor(states)
            true_value_tensor = torch.tensor(true_values)
            for ind, s in enumerate(states):
                disc_state = self.discretize_state(s, bins_per_dimension)
                mat_counts[i, int(disc_state[0] - 1),int(disc_state[1] - 1)] = true_value_tensor[ind]
        return mat_counts

    def get_Q_abs_diff(self, global_step, state_to_vals):
        q_mat = self.get_discretized_value_abs_diff(self.duel_network, state_to_vals)
        self.plot_matrix(q_mat[0], "AbsDiffQ1", global_step)
        self.plot_matrix(q_mat[1], "AbsDiffQ2", global_step)
        self.plot_matrix(np.mean(q_mat, axis=0), "AbsDiffMeanQ", global_step)

    def plot_Q_func(self, global_step):
        q_mat = self.get_discretized_value_mat(self.duel_network, q=True)
        self.plot_matrix(q_mat[0], "Q1", global_step)
        self.plot_matrix(q_mat[1], "Q2", global_step)
        self.plot_matrix(np.mean(q_mat, axis=0), "MeanQ", global_step)

    def plot_true_Q(self, global_step, state_to_vals):
        q_mat = self.get_discretized_true_q(state_to_vals)
        self.plot_matrix(q_mat[0], "TrueQ1", global_step)
        self.plot_matrix(q_mat[1], "TrueQ2", global_step)
        self.plot_matrix(np.mean(q_mat, axis=0), "TrueMeanQ", global_step)

    def get_trajectories_beh_agent(self, global_step, num_trajectories=4):
        # computes trajectories of beh policy
        start_locations = [[0.1, 0.1], [1., 0.1], [0.5, 0.5], [0.,0.9]]
        for traj in range(num_trajectories):
            trajectory = []
            obs, _ = self.eval_env.reset(options= {"x":start_locations[traj][0], "y":start_locations[traj][1]})
            done = False
            max_steps = 200
            curr_step = 0
            while not done and curr_step <= max_steps:
                trajectory.append(obs)
                actions = self.select_behavior_action(obs, epsilon=0, var_lambda=0)
                next_obs, rewards, terminations, truncations, _ = self.eval_env.step(actions[0])
                done = terminations or truncations
                obs = next_obs
                curr_step += 1
            trajectory = np.vstack(trajectory)
            plt, fig = self.eval_env.plot_trajectory(trajectory)
            # Instead of directly logging the figure to WandB, save it to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)

            # Load the image from the buffer into PIL
            image = Image.open(buf)

            # Convert the image to RGB (if it's not already in RGB)
            image = image.convert('RGB')

            # Convert the image to a numpy array
            image_array = np.array(image)

            # Convert the numpy array to PyTorch tensor
            # Permute the dimensions to [C, H, W] as expected by TensorBoard
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

            # Use the SummaryWriter to add the figure
            self.writer.add_image(f'traj_{global_step}/Trajectory', image_tensor, global_step)
            buf.close()


    def get_target_policy_prob(self, obs, target_policy):
        if self.args.env_id in ["RoomMultiGoals-v0", "RoomSingleGoals-v0", "MultiVariedGoalRoomEnv-v0"]:
            action_probs = target_policy.get_action_prob(torch.Tensor(obs)).to(self.device)
            return action_probs


    def get_all_target_probs(self, obs, target_policies):
        target_action_probs = []
        for i in range(self.num_target_policies):
            target_action_probs.append(self.get_target_policy_prob(obs, target_policies[i]))
        target_action_probs = torch.stack(target_action_probs, dim=-1)  # [n * a * r]
        return target_action_probs

    def get_behavior_policy_prob(self, obs, target_policies, var_lambd):
        min_val = 1e-5
        max_val = 1e8

        with torch.no_grad():
            var_vals = self.duel_network(torch.Tensor(obs), update_var_only=True).to(self.device) # [n * a * r]
            target_action_probs = self.get_all_target_probs(obs, target_policies) # [n * a * r]
            # if var_vals == 0:
            #     print(f"zero val of variance at obs {obs}!!!!!")

            var_vals = torch.clamp(var_vals, min=1e-4) # 0.01

            # numerator = torch.sqrt(torch.einsum('nar,nar->na', torch.square(target_action_probs), var_vals)) # [na] dim

            if var_lambd>0:
                numerator = var_vals * (1-var_lambd) + var_lambd* (torch.ones_like(var_vals)*self.infty_approx)
            else:
                numerator = var_vals

            numerator = torch.sqrt(torch.einsum('nar,nar->na', torch.square(target_action_probs), numerator))  # [na] dim

            # print(f" var lambda {var_lambd}")
            # numerator = torch.clamp(numerator, min=min_val, max=(max_val/10))
            denominator = torch.sum(numerator, dim=1, keepdim=True)
            # Ensure denominator does not contain infinity.
            denominator = torch.clamp(denominator, min=min_val, max=max_val)

            # print(f"numerator dim: {numerator.shape} and denomiantor shape: {denominator.shape} and denom: {denominator}")

            behavior_policy_action_prob = numerator/denominator # [na]
            behavior_policy_action_prob = torch.clamp(behavior_policy_action_prob, min=0.01) #0.01
            behavior_policy_action_prob = behavior_policy_action_prob / behavior_policy_action_prob.sum(dim=-1, keepdim=True)

            # print("beh policy action prob:", behavior_policy_action_prob)

            # Check for any infinities or NaNs in behavior_policy after the computation.
            if torch.isinf(behavior_policy_action_prob).any() or torch.isnan(behavior_policy_action_prob).any():
                raise ValueError("behavior_policy contains infinity or NaN values")

        return behavior_policy_action_prob # [n X a] dim

    def check_weight_change(self, weights_before, net):
        # Check if weights have changed
        weights_have_changed = False
        for name, param in net.named_parameters():
            if not torch.equal(weights_before[name], param):
                weights_have_changed = True
                break
        return weights_have_changed


    def train_var_agent(self, global_step):
        if self.args.prioritized_replay:
            data = self.rb.sample(self.args.batch_size, beta=self.beta_schedule.value(global_step))
        else:
            data = self.rb.sample(self.args.batch_size)

        # set training modes
        # self.turn_eval_mode()
        # self.var_network.train()

        # Store a copy of the weights
        # q_net_weights_before = self.get_weights_copy(self.q_network)
        with torch.no_grad():
            # construct td error for q
            target_q_vals = self.target_duel_network(data.next_observations, update_q_only=True)  # [batch size * a dim * reward dim]

            if self.args.type_next_target_val == 0:
                target_val =self.get_exp_v_target(target_q_vals, data.next_observations)
            elif self.args.type_next_target_val == 1:
                target_val = self.get_rho_corrected_q_target(target_q_vals, data.next_observations, data.next_actions)

            # target_v_vals = self.get_v_val(target_q_vals, data.next_observations)  # # [batch size * reward dim]
            curr_dones = (1 - data.dones).reshape(-1)
            old_val = self.duel_network(data.observations, update_q_only=True)
            actions = data.actions
            actions_expanded = actions.unsqueeze(-1).expand(-1, -1, old_val.size(-1))
            old_val = torch.gather(old_val, 1, actions_expanded)
            old_val = old_val.squeeze(1)
            td_error_q = data.rewards + args.gamma * torch.einsum('n,nr->nr', curr_dones, target_val) - old_val #[nr] dim

            # construct td error for var
            target_var_vals = self.target_duel_network(data.next_observations, update_var_only=True)  # [batch size * a dim * reward dim]
            if self.args.type_next_target_val == 0:
                target_val_new = self.get_exp_v_target(target_var_vals, data.next_observations)
            elif self.args.type_next_target_val == 1:
                target_val_new = self.get_rho_corrected_var_target(target_var_vals, data.next_observations, data.next_actions)

            # target_var_v_vals = self.get_v_val(target_var_vals, data.next_observations)  # # [batch size * reward dim]
            curr_dones = (1 - data.dones).reshape(-1)
            td_var_target = torch.square(td_error_q) + (args.gamma**2) * torch.einsum('n,nr->nr', curr_dones, target_val_new)
            # print(" target shape in var:", td_var_target.shape)

        old_var_val = self.duel_network(data.observations, update_var_only=True)
        actions = data.actions
        actions_expanded = actions.unsqueeze(-1).expand(-1, -1, old_var_val.size(-1))
        old_var_val = torch.gather(old_var_val, 1, actions_expanded)
        old_var_val = old_var_val.squeeze(1)
        if self.args.var_type_loss == 0:
            var_loss = torch.abs(td_var_target - old_var_val)
        elif self.args.var_type_loss == 1:
            var_loss = (td_var_target - old_var_val) ** 2.

        # compute priority for PER buffer
        if self.args.prioritized_replay:
            with torch.no_grad():
                # Calculate TD error
                td_error_var = torch.abs(td_var_target - old_var_val.detach()).mean(dim=1)
            # Convert TD error to numpy array and flatten (if necessary)
            new_priority = td_error_var.cpu().numpy().flatten() + self.args.prioritized_replay_eps
            self.rb.update_priorities(data.indices, new_priority)
            # correct for IS by using the weights
            loss = torch.einsum('nr,n->nr', var_loss, torch.tensor(data.weights))

        var_loss = var_loss.mean()
        # optimize the model
        self.optimizer.zero_grad()
        var_loss.backward()
        self.optimizer.step()

        # check for q network weight change
        # weights_changed = self.check_weight_change(q_net_weights_before, self.q_network)
        # print(f"q network weight change at {global_step} is: {weights_changed}")

        # update target network
        if global_step % self.args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_duel_network.parameters(),
                                                             self.duel_network.parameters()):
                target_network_param.data.copy_(
                    self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                )
            self.target_duel_network.eval()

        if global_step % 100 == 0:
            printable_old_val = old_var_val.mean(axis=0).detach().cpu().numpy()
            for i in range(len(printable_old_val)):
                # Log each dimension separately
                self.writer.add_scalar(f"losses/var_values_{i + 1}", printable_old_val[i], global_step)
            self.writer.add_scalar("losses/td_loss_var_mean", var_loss.mean().item(), global_step)
            self.writer.add_scalar("losses/var_values_mean", old_var_val.mean().item(), global_step)


    def get_rho_corrected_q_target(self, q_val, obs, action):
        # print(f"In RHo function-------next actions: {action}")
        with torch.no_grad():
            rho, pi, behv = self.get_rho_pi_b(obs)
            rho_prod_q = rho * q_val # [nar] dim
            actions_expanded = action.unsqueeze(-1).expand(-1, -1, rho.size(-1))
            rho_prod_q = torch.gather(rho_prod_q, 1, actions_expanded) #
            corrected_q_val = rho_prod_q.squeeze(1) # [nr] dim
            # print(f"rho shape: {rho.shape}, rho: {rho}, next val: {corrected_q_val.shape}")
        return corrected_q_val

    def get_rho_corrected_var_target(self, var_val, obs, action):
        with torch.no_grad():
            rho, pi, behv = self.get_rho_pi_b(obs)
            rho_prod_var = torch.square(rho) * var_val # [nar] dim
            actions_expanded = action.unsqueeze(-1).expand(-1, -1, rho.size(-1))
            rho_prod_var = torch.gather(rho_prod_var, 1, actions_expanded) #
            corrected_var_val = rho_prod_var.squeeze(1) # [nr] dim
        return corrected_var_val

    def get_exp_v_target(self, q_val, obs):
        with torch.no_grad():
            target_action_probs = self.get_all_target_probs(obs, self.target_policy_agents)
            q_vals = q_val.reshape(-1, self.action_dim, self.reward_dim)
            target_v_vals = torch.einsum('nar,nar->nr', target_action_probs, q_vals)
        return target_v_vals #[nr] dim

    def get_rho_pi_b(self, obs):
        with torch.no_grad():
            # target policy prob
            target_action_probs = self.get_all_target_probs(obs, self.target_policy_agents) # [nar]
            # behavior policy prob
            behavior_policy_prob = self.get_behavior_policy_prob(obs, self.target_policy_agents, self.var_lambda) # [na]
            behavior_policy_prob = behavior_policy_prob.unsqueeze(-1)
            # rho
            rho = target_action_probs/behavior_policy_prob # [nar] dim
            rho = torch.clamp(rho, max=1.)
        return rho, target_action_probs, behavior_policy_prob # all [nar] dim


    def train_q_agent(self, global_step):
        if self.args.prioritized_replay:
            data = self.rb.sample(self.args.batch_size, beta=self.beta_schedule.value(global_step))
        else:
            data = self.rb.sample(self.args.batch_size)

        # set training modes

        with torch.no_grad():
            target_q_vals = self.target_duel_network(data.next_observations, update_q_only=True) # [batch size * a dim * reward dim]

            if self.args.type_next_target_val == 0:
                target_val =self.get_exp_v_target(target_q_vals, data.next_observations)
            elif self.args.type_next_target_val == 1:
                target_val = self.get_rho_corrected_q_target(target_q_vals, data.next_observations, data.next_actions)

            # target_v_vals = self.get_v_val(target_q_vals, data.next_observations) # # [batch size * reward dim]
            curr_dones = (1 - data.dones).reshape(-1)
            td_target = data.rewards + self.args.gamma * torch.einsum('n,nr->nr', curr_dones, target_val) # [nr] dim
            # print(" target shape:", td_target.shape)

        old_val = self.duel_network(data.observations, update_q_only=True) # [nar] dim
        actions = data.actions
        # print("action shape:", actions)
        actions_expanded = actions.unsqueeze(-1).expand(-1, -1, old_val.size(-1))
        # print(f"new and shape {actions_expanded.shape} and actions {actions_expanded}")
        old_val = torch.gather(old_val, 1, actions_expanded)
        old_val = old_val.squeeze(1) # [nr] dim


        if self.args.q_type_loss == 0:
            loss = torch.abs(td_target-old_val)
        elif self.args.q_type_loss == 1:
            loss = (td_target - old_val)**2.

        if self.args.prioritized_replay:
            loss = torch.einsum('nr,n->nr', loss, torch.tensor(data.weights))


        loss = loss.mean()

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if global_step % self.args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_duel_network.parameters(), self.duel_network.parameters()):
                target_network_param.data.copy_(
                    self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                )
            self.target_duel_network.eval()

        if global_step % 100 == 0:
            printable_old_val = old_val.mean(axis=0).detach().cpu().numpy()
            for i in range(len(printable_old_val)):
                # Log each dimension separately
                self.writer.add_scalar(f"losses/q_values_{i+1}", printable_old_val[i], global_step)
            self.writer.add_scalar("losses/td_loss_q_mean", loss.mean().item(), global_step)
            self.writer.add_scalar("losses/q_values_mean", old_val.mean().item(), global_step)

    def joint_training(self, global_step):
        if self.args.prioritized_replay:
            data = self.rb.sample(self.args.batch_size, beta=self.beta_schedule.value(global_step))
        else:
            data = self.rb.sample(self.args.batch_size)

        # set training modes

        with torch.no_grad():
            target_q_vals = self.target_duel_network(data.next_observations, update_q_only=True) # [batch size * a dim * reward dim]
            target_var_vals = self.target_duel_network(data.next_observations,
                                                     update_var_only=True)  # [batch size * a dim * reward dim]
            if self.args.type_next_target_val == 0:
                target_val =self.get_exp_v_target(target_q_vals, data.next_observations)
                target_var_val = self.get_exp_v_target(target_var_vals, data.next_observations)
            elif self.args.type_next_target_val == 1:
                target_val = self.get_rho_corrected_q_target(target_q_vals, data.next_observations, data.next_actions)
                target_var_val = self.get_rho_corrected_q_target(target_var_vals, data.next_observations, data.next_actions)

            # target_v_vals = self.get_v_val(target_q_vals, data.next_observations) # # [batch size * reward dim]
            curr_dones = (1 - data.dones).reshape(-1)
            td_target_val = data.rewards + self.args.gamma * torch.einsum('n,nr->nr', curr_dones, target_val) # [nr] dim

            # print(" target shape:", td_target.shape)

        old_val = self.duel_network(data.observations, update_q_only=True) # [nar] dim
        old_var = self.duel_network(data.observations, update_var_only=True)  # [nar] dim

        actions = data.actions
        actions_expanded = actions.unsqueeze(-1).expand(-1, -1, old_val.size(-1))

        old_val = torch.gather(old_val, 1, actions_expanded)
        old_val = old_val.squeeze(1) # [nr] dim

        old_var = torch.gather(old_var, 1, actions_expanded)
        old_var = old_var.squeeze(1)  # [nr] dim

        td_error = td_target_val - old_val

        with torch.no_grad():
            td_target_var = td_error**2 + (self.args.gamma**2) * torch.einsum('n,nr->nr', curr_dones, target_var_val) # [nr] dim

        val_loss = (td_target_val - old_val) **2.
        var_loss = (td_target_var - old_var) **2.

        loss = 0.75*var_loss + val_loss

        if self.args.prioritized_replay:
            loss = torch.einsum('nr,n->nr', loss, torch.tensor(data.weights))

        loss = loss.mean()

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if global_step % self.args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_duel_network.parameters(), self.duel_network.parameters()):
                target_network_param.data.copy_(
                    self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                )
            self.target_duel_network.eval()

        if global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss_mean", loss.mean().item(), global_step)
            self.writer.add_scalar("losses/q_values_mean", old_val.mean().item(), global_step)
            self.writer.add_scalar("losses/var_values_mean", old_var.mean().item(), global_step)

    def train_agent(self, global_step):
        # TODO: COMPLETE THE Q AND VAR CALLS
        self.joint_training(global_step)
        #
        # self.train_q_agent(global_step)
        #
        # if global_step % self.args.variance_train_frequency == 0:
        #     self.train_var_agent((global_step))


    def save_checkpoint(self, global_step):
        """
        Saves the state of the model and optimizer at a particular global step.

        :param model: The Q-network model (an instance of nn.Module)
        :param optimizer: The optimizer used for training
        :param global_step: The current global step in training
        :param directory: Directory where to save the checkpoint
        :param filename: Base filename of the checkpoint
        """
        checkpoint_dir = os.path.join(self.args.run_save_loc, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, f"{global_step}.pt")

        torch.save({
            'global_step': global_step,
            'state_dict': self.duel_network.state_dict(),
            'target_state_dict': self.target_duel_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)

        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, global_step):
        """
        Saves the state of the model and optimizer at a particular global step.

        :param model: The Q-network model (an instance of nn.Module)
        :param optimizer: The optimizer used for training
        :param global_step: The current global step in training
        :param directory: Directory where to save the checkpoint
        :param filename: Base filename of the checkpoint
        """
        checkpoint_path = os.path.join(self.args.run_save_loc, "checkpoint", f"{global_step}.pt")
        if not os.path.exists(checkpoint_path):
            print("checkpoint file does not exist:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.duel_network.load_state_dict(checkpoint['state_dict'])
        self.target_duel_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"checkpoint loaded at global step {global_step}")

    def save_model(self):
        model_save_path = os.path.join(self.args.run_save_loc, "saved_model")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            print("model path created", model_save_path)
        torch.save(self.duel_network.state_dict(), os.path.join(model_save_path, "model.pth"))
        print(f"model saved to {model_save_path}")

    def load_model(self):
        model_path = os.path.join(self.args.run_save_loc, "saved_model", "model.pth")
        self.duel_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.duel_network.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"model loaded from {model_path}")


    def evaluate_agent_perf(self, global_step):
        # print("evaluatin code......")
        evaluate_episodic_return = []
        evaluate_episodic_len = []
        self.turn_eval_mode()
        for eps in range(self.args.eval_episodes):
            total_reward = np.zeros(self.reward_dim)
            eps_len = 0
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                eps_len +=1
                action = self.select_action(torch.Tensor(obs), epsilon=0.)
                next_obs, rewards, terminations, truncations, infos = self.eval_env.step(action[0])
                total_reward +=rewards
                done = truncations or terminations
                obs = next_obs
            evaluate_episodic_return.append(total_reward)
            evaluate_episodic_len.append(eps_len)
        evaluate_episodic_return_arr= np.array(evaluate_episodic_return)
        evaluate_episodic_len_arr = np.array(evaluate_episodic_len)
        mean_return = np.mean(evaluate_episodic_return_arr, axis=0)
        mean_len = np.mean(evaluate_episodic_len_arr)
        # print(f"mean return shape in evalate {mean_return.shape}")
        for dim in range(mean_return.shape[-1]):
            self.writer.add_scalar(f"eval_perf/episodic_return_{dim+1}", mean_return[dim], global_step)
        self.writer.add_scalar(f"eval_perf/episodic_return_avg", np.mean(mean_return), global_step)
        self.writer.add_scalar(f"eval_perf/episodic_len", mean_len, global_step)

    def evaluate_mse_agent(self, global_step, state_to_true_val_dict_list):
        # evaluates the avg MSE
        self.turn_eval_mode()
        mse_loss = []
        for i in range(self.reward_dim):
            state_to_val = state_to_true_val_dict_list[i]
            states, values = zip(*state_to_val.items())
            state_tensor = torch.tensor(states)
            true_value_tensor = torch.tensor(values)
            with torch.no_grad():
                estimated_q_vals = self.duel_network(state_tensor, update_q_only=True).to(self.device) #nar
                estimated_v_vals = self.get_exp_v_target(estimated_q_vals, state_tensor)[:, i] #nr
                estimated_v_vals = estimated_v_vals.reshape(-1)
                mse_loss.append(F.mse_loss(estimated_v_vals, true_value_tensor).item())

        for dim in range(self.reward_dim):
            self.writer.add_scalar(f"eval_mse/mse_{dim + 1}", mse_loss[dim], global_step)
            self.writer.add_scalar(f"eval_mse/rmse_{dim + 1}", np.sqrt(mse_loss[dim]), global_step)
        self.writer.add_scalar(f"eval_mse/mse_avg", np.mean(mse_loss), global_step)
        self.writer.add_scalar(f"eval_mse/rmse_avg", np.mean(np.sqrt(mse_loss)), global_step)

    def turn_eval_mode(self):
        self.duel_network.eval()

def load_target_policy_and_state_val_dict(args):
    args.track = False
    # saved models location

    # rooms target policies
    loc_1 = "/home/mila/j/jainarus/scratch/multi_varied_goal_room/goal_0"
    loc_2 = "/home/mila/j/jainarus/scratch/multi_varied_goal_room/goal_1"

    # loading agents
    args.puddle_goal_type = 0
    agent_1 = SimpleTargetPolicy(args)

    args.puddle_goal_type = 1
    agent_2 = SimpleTargetPolicy(args)


    target_agents = [agent_1, agent_2]
    print("Target policies loaded!")

    # load state_to_val dic
    # load state_to_val dic
    if args.compute_mse_for_random_states == 1:
        state_val_loc = "combined_true_val_random_states"
    else:
        state_val_loc = "combined_true_val_pi_states"

    state_to_val_1 = load_state_to_val_dict(os.path.join(loc_1, state_val_loc, "state_to_val.json"))
    state_to_val_2 = load_state_to_val_dict(os.path.join(loc_2, state_val_loc, "state_to_val.json"))  # dictionary
    state_to_vals = [state_to_val_1, state_to_val_2]
    print("State to val dict loaded!")

    return target_agents, state_to_vals

def safe_exit(target_agents, agent):
    agent.env.close()
    agent.eval_env.close()
    agent.writer.close()


if __name__ == "__main__":
    args = parse_args()
    args_target_copy = copy.deepcopy(args)

    target_policies_agents, state_to_vals = load_target_policy_and_state_val_dict(args_target_copy)
    running_agent = VariancePolicyEvaluationAgent(args, target_policies_agents)

    plot_q_true_func = 1

    for global_step in range(args.total_timesteps):
        running_agent.collect_rollout(global_step) # get data for buffer
        if plot_q_true_func == 1:
            running_agent.plot_true_Q(global_step, state_to_vals)
            plot_q_true_func=0

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                running_agent.train_agent(global_step) # train the network

            if global_step % args.eval_frequency == 0:
                # running_agent.evaluate_agent_perf(global_step) # evaluate the network perf
                running_agent.evaluate_mse_agent(global_step, state_to_vals)  # evaluate for mse calculation

            if global_step % args.beh_plot_frequency == 0:
                # plotting beh policy trajectories
                running_agent.get_trajectories_beh_agent(global_step, num_trajectories=4)
                # get state visitation count
                running_agent.get_trajectories_for_state_visitation_beh_policy(global_step, num_trajectories=1000)
                # plot var
                running_agent.plot_variance_func(global_step)
                # plot q abs difference
                running_agent.get_Q_abs_diff(global_step, state_to_vals)
                #plot estimate Q
                running_agent.plot_Q_func(global_step)



            if global_step % args.checkpoint_frequency == 0:
                running_agent.save_checkpoint(global_step)

    running_agent.get_trajectories_beh_agent(args.total_timesteps, num_trajectories=4)  # plotting beh policy trajctories

    running_agent.save_model()
    safe_exit(target_policies_agents, running_agent)


