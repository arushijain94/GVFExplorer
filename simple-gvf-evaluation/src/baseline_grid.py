import argparse
import envs
import os
import random
import time
from distutils.util import strtobool
import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.common import MultiRewardReplayBuffer, MultiRewardNextActionReplayBuffer, PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from src.common import load_config, linear_schedule, make_env, load_state_to_val_dict, save_temp_config, create_env_args, LinearScheduleClass, discretize_state, plot_matrix
import wandb
import warnings
from src.multi_dim_dqn import DQNAgent
from src.simple_target_policy import SimpleTargetPolicy
import copy
import yaml
import sys
import io
from PIL import Image
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="the group (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--run-save-loc", type=str, default="",
                        help="save location for the data")


    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--goal-position", type=float, default=0.0, help="position of goal for pendulum np.pi") # goal position for target_policies, don't modify the yml for this!
    parser.add_argument("--goal-positions-pe", type=float, nargs=2, default=[0.0, -0.78],
                        help="Two positions of goal for pendulum, separated by a space") # goal position for PE
    parser.add_argument("--puddle-goal-type", type=int, default=0,
                        help="{0: [0.95, 0.95], 1: [0.4, 0.65], 2: , 3: both goals")

    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-step", type=int, default=5000,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--eval-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="the episodes of evaluation")
    parser.add_argument("--compute-mse-for-random-states", type=int, default=1,
                        help="{0: pi state dist, 1: random states}")
    parser.add_argument("--type-target-policy", type=int, default=1,
                        help="[1: round robin, 2: mixture, 3: uniform]")
    parser.add_argument(
        "--plot-frequency", type=int, default=1000, help="store plot")
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

    # Parse remaining command line arguments
    args = parser.parse_args(remaining_argv)
    return args

class PolicyEvaluationAgent:
    # target_policy_agents: list of target policy agents.

    def __init__(self, args, target_policy_agents):
        self.args =args
        self.num_target_policies = len(target_policy_agents)
        self.target_policy_agents = target_policy_agents # target policy agent networks list

        run_name = f"{args.env_id}__{self.args.wandb_group}__LR{self.args.learning_rate}__Seed{self.args.seed}"
        self.args.run_name = run_name

        current_time = datetime.datetime.now().strftime("%m%d_%H%M")
        wandb_dir = f"{self.args.wandb_group}_{current_time}"
        wandb_dir = f"{self.args.run_save_loc}/wandb/{wandb_dir}"
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)

        self.args.run_save_loc = os.path.join(self.args.run_save_loc, run_name)
        # print("new locations in PE: *******", self.args.run_save_loc)
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
                dir=wandb_dir,
            )

        self.writer = SummaryWriter(f"{self.args.run_save_loc}/runs")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        # env setup
        env_args = create_env_args(args, args.env_id)
        self.env = make_env(self.args.env_id, self.args.seed, self.args.capture_video, run_name, env_args) # training
        self.eval_env = make_env(self.args.env_id, self.args.seed, self.args.capture_video, run_name, env_args)

        self.observation_space = self.env.observation_space
        self.state_dim = self.observation_space.n
        print("state dim:", self.state_dim)
        self.action_space = self.env.action_space
        self.action_dim = self.action_space.n
        self.reward_dim = self.env.reward_space.n
        self.true_q = self.load_true_q()
        assert isinstance(self.action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.q = np.zeros((self.state_dim, self.action_dim, self.reward_dim))
        self.var = np.zeros((self.state_dim, self.action_dim, self.reward_dim))
        self.epsilon_schedule = LinearScheduleClass(self.args.exploration_step,
                                       initial_p=self.args.start_e,
                                       final_p=self.args.end_e)


        # reset env
        self.obs, _ = self.env.reset()
        self.current_target_policy_index = 0
        self.train_episodic_return = np.zeros(self.reward_dim)
        self.train_episodic_len = 0
        self.action = None

        # save config
        save_temp_config(self.args.run_save_loc, self.args)

    def get_expected_val(self, val, pol):
        return np.einsum('ar,ar->r', val, pol)

    def update_q(self, state, action, next_state, rewards):
        td_error = rewards + self.args.gamma * \
                   self.get_expected_val(self.q[next_state], self.target_policy_agents[next_state])\
                   - self.q[state, action]
        self.q[state, action] = self.q[state, action] + self.args.learning_rate * td_error
        return td_error

    def update_var(self, state, action, next_state, q_td_error):
        var_td_error = np.square(q_td_error) + self.args.gamma**2 *\
                       self.get_expected_val(self.var[next_state], self.target_policy_agents[next_state]) \
                       - self.var[state]
        self.var[state, action] = self.var[state, action] + self.args.learning_rate * var_td_error

    def write_metrics(self, global_step):
        self.plot_val(global_step)
        self.plot_var(global_step)
        self.plot_abs_q_diff()

    def train(self, epsilon, num_episodes=5000):
        global_step = 0
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                # Choose action based on epsilon-greedy policy
                action = self.select_action(obs,epsilon)
                next_obs, rewards, terminations, truncations, infos = self.env.step(action)
                self.update_q(obs, action, next_obs)
                self.update_var(obs, action, next_obs)
                obs = next_obs
                done = terminations or truncations
                global_step +=1

                if global_step % 100 == 0:
                    self.write_metrics(global_step)

    def select_action(self, obs, epsilon, plot_policy_index=False,
                      policy_index=0):
        # for single observation
        if random.random() < epsilon:
            actions = np.array([self.env.action_space.sample()])
        else:
            with torch.no_grad():
                if self.args.type_target_policy == 1: # round robin
                    if plot_policy_index:
                        current_target_policy_agent = self.target_policy_agents[policy_index]
                    else:
                        current_target_policy_agent = self.target_policy_agents[
                            self.current_target_policy_index]  # select target policy net
                    action_probs = self.get_target_policy_prob(obs, current_target_policy_agent)
                if self.args.type_target_policy == 2: # mixture
                    action_prob_mean = np.stack([self.get_target_policy_prob(obs, self.target_policy_agents[i])
                                                    for i in range(self.num_target_policies)]).mean(dim=0)
                    action_probs = action_prob_mean / action_prob_mean.sum()
                if self.args.type_target_policy == 3: # uniform policy
                    return np.array([self.env.action_space.sample()])
                actions = np.sample.choice(np.arange(self.action_dim), p=action_probs)
        return actions

    def collect_rollout(self, global_step):
        # 1 step execution of env
        epsilon = self.epsilon_schedule.value(global_step)
        if self.action is None:
            self.action = self.select_action(self.obs, epsilon) # select action from target policy agent
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = self.env.step(self.action[0])
        self.train_episodic_return += rewards
        self.train_episodic_len +=1
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        dones = terminations or truncations

        if dones is True:
            next_actions = np.zeros_like(self.action)
        else:
            next_actions = self.select_action(real_next_obs, epsilon)
        real_next_action = next_actions.copy()

        self.rb.add(self.obs, real_next_obs, self.action, real_next_action, rewards, dones, infos)

        if dones:
            self.obs, _ = self.env.reset()
            self.action = None
            # switch the target policy index
            self.current_target_policy_index = (self.current_target_policy_index + 1) % self.num_target_policies
            # print(f" policy index: {self.current_target_policy_index}")
            self.train_episodic_return = np.zeros(self.reward_dim)
            self.train_episodic_len = 0
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            self.obs = next_obs
            self.action = next_actions

    def get_target_policy_prob(self, obs, target_policy):
        # print("shape pf pbs in target:", torch.Tensor(obs).size())
        if self.args.env_id in ["RoomMultiGoals-v0", "RoomSingleGoals-v0",
                                "MultiVariedGoalRoomEnv-v0", "GVFEnv-v0", "FourRoomsEnv-v0"]:
            action_probs = target_policy.get_action_prob(torch.Tensor(obs).to(self.device)).to(self.device)
            return action_probs

    def get_v_val(self, q_vals, obs):
        target_action_probs = []
        for i in range(self.reward_dim):
            target_action_probs.append(self.get_target_policy_prob(obs, self.target_policy_agents[i]))
        target_action_probs = np.stack(target_action_probs, axis=-1) # [a * r] action probabilities pi(a|s)
        q_vals = q_vals.reshape(-1, self.action_dim, self.reward_dim)
        v_vals = torch.einsum('nar,nar->nr', target_action_probs, q_vals)
        return v_vals


    def train_agent(self, global_step):
        if self.args.prioritized_replay:
            data = self.rb.sample(self.args.batch_size, beta=self.beta_schedule.value(global_step))
        else:
            data = self.rb.sample(self.args.batch_size)
        # set training modes
        self.q_network.train()
        self.target_network.eval()

        with torch.no_grad():
            target_q_vals = self.target_network(data.next_observations) # [batch size * a dim * reward dim]
            target_v_vals = self.get_v_val(target_q_vals, data.next_observations) # # [batch size * reward dim]
            curr_dones = (1 - data.dones).reshape(-1)
            td_target = data.rewards + args.gamma * torch.einsum('n,nr->nr', curr_dones, target_v_vals)
            # print(" target shape:", td_target.shape)

        old_val = self.q_network(data.observations)
        # Expand actions to match old_val's dimensions
        actions = data.actions
        actions_expanded = actions.unsqueeze(-1).expand(-1, -1, old_val.size(-1))
        old_val = torch.gather(old_val, 1, actions_expanded)
        old_val = old_val.squeeze(1)


        if self.args.q_type_loss == 0:
            loss = torch.abs(td_target - old_val)
        elif self.args.q_type_loss == 1:
            loss = (td_target - old_val)**2.

        # compute priority for PER buffer
        if self.args.prioritized_replay:
            with torch.no_grad():
                # Calculate TD error
                td_error = torch.abs(td_target - old_val.detach()).mean(dim=1)  # Assuming mean TD error over reward_dim
            # Convert TD error to numpy array and flatten (if necessary)
            new_priority = td_error.cpu().numpy().flatten() + self.args.prioritized_replay_eps
            # print("update priority:", new_priority)
            self.rb.update_priorities(data.indices, new_priority)
            # correct for IS by using the weights
            # print(f"old loss: {loss}, and weights {torch.tensor(data.weights).shape}")
            loss = torch.einsum('nr,n->nr', loss, torch.tensor(data.weights))
            # print("new loss:", loss)

        loss = loss.mean()
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if global_step % self.args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                )
            self.target_network.eval()

        if global_step % 100 == 0:
            printable_old_val = old_val.mean(axis=0).detach().cpu().numpy()
            for i in range(len(printable_old_val)):
                # Log each dimension separately
                self.writer.add_scalar(f"losses/q_values_{i+1}", printable_old_val[i], global_step)
            # self.writer.add_scalar("losses/td_loss_q_mean", loss.mean().item(), global_step)
            self.writer.add_scalar("losses/q_values_mean", old_val.mean().item(), global_step)

    def discretize_state(self, state, bins_per_dimension):
        discrete_state = tuple(np.digitize(s, b) for s, b in zip(state, bins_per_dimension))
        return discrete_state

    def get_discretized_value_mat(self, net, state_to_true_val_dict_list, bins=20):
        mat_counts = np.zeros((self.reward_dim, bins, bins))
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                                   zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]

        for i in range(self.reward_dim):
            state_to_val = state_to_true_val_dict_list[i]
            states, true_values = zip(*state_to_val.items())
            state_tensor = torch.tensor(states)
            true_value_tensor = torch.tensor(true_values)
            with torch.no_grad():
                estimated_s_a_vals = net(state_tensor).to(self.device)
                estimated_v_vals = self.get_v_val(estimated_s_a_vals, state_tensor)[:, i]
                estimated_v_vals = estimated_v_vals.reshape(-1)
                abs_diff_error = torch.abs(estimated_v_vals - true_value_tensor)
            abs_diff_mat = abs_diff_error.cpu().detach().numpy()
            for ind, s in enumerate(states):
                disc_state = self.discretize_state(s, bins_per_dimension)
                mat_counts[i, int(disc_state[0] - 1),int(disc_state[1] - 1)] = abs_diff_mat[ind]
        return mat_counts

    def get_discretized_est_q(self, net, num_samples=5, bins=20):
        mat_counts = np.zeros((self.reward_dim, bins, bins))
        bin_size = 1./bins
        for i in range(0, bins):
            for j in range(0, bins):
                start_x = i*bin_size
                start_y = j*bin_size
                data = np.random.uniform((start_x,start_y), (start_x+bin_size, start_y+bin_size), (num_samples, 2))
                data = np.clip(data, a_min=0., a_max=1.)
                with torch.no_grad():
                    val = net(torch.Tensor(data).to(self.device)).mean(dim=(0,1)) #r
                    mat_counts[:, i, j] = val.cpu().detach().numpy()
        return mat_counts

    def get_trajectories_for_state_visitation_policy_target(self, global_step, bins=20, num_trajectories=1000):
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                              zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]

        visitation_counts = np.zeros((bins, bins))
        pol_index=0
        for traj in range(num_trajectories):
            obs, _ = self.eval_env.reset()
            done = False
            # max_steps = 200
            curr_step = 0
            while not done:
                actions = self.select_action(obs, epsilon=0, plot_policy_index=True, policy_index=pol_index)
                next_obs, rewards, terminations, truncations, _ = self.eval_env.step(actions[0])
                # Discretize the state and update visitation counts
                discretized_state = discretize_state(obs, bins_per_dimension)
                visitation_counts[discretized_state[0]-1][discretized_state[1]-1] += 1
                done = terminations or truncations
                obs = next_obs
                curr_step += 1
            # print(f"traj done {traj}")
            pol_index = (pol_index+1)%self.reward_dim
        # print(f"done get traj")
        plt, fig = self.eval_env.plot_mat(visitation_counts, normalized_counts=True)
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
        self.writer.add_image(f'visitation_{global_step}', image_tensor, global_step)
        buf.close()

    def get_trajectories_agent(self, global_step, num_trajectories=4, policy_index=0):
        # computes trajectories of beh policy
        start_locations = [[0.1, 0.1], [1., 0.1], [0.5, 0.5], [0.,0.9]]
        for traj in range(num_trajectories):
            trajectory = []
            obs, _ = self.eval_env.reset(options= {"x":start_locations[traj][0], "y":start_locations[traj][1]})
            done = False
            max_steps = 500
            curr_step = 0
            while not done and curr_step <= max_steps:
                trajectory.append(obs)
                actions = self.select_action(obs, epsilon=0, plot_policy_index=True, policy_index=policy_index)
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
            self.writer.add_image(f'traj_{global_step}_loc{traj}_pol{policy_index}', image_tensor, global_step)
            buf.close()

    def get_discretized_true_q(self, state_to_true_val_dict_list, bins=20):
        mat_counts = np.zeros((self.reward_dim, bins, bins))
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


    def plot_val(self, global_step):
        # print("in plot v val difference")
        v_mat = np.einsum('sar,sar->sr', self.q, self.target_policy_agents)
        for i in range(self.reward_dim):
                plot_matrix(self.eval_env, self.writer, v_mat[i], f"V_{i+1}", global_step)
        plot_matrix(self.eval_env, self.writer, np.mean(v_mat, axis=-1), "MeanV", global_step)

    def plot_var(self, global_step):
        # print("in plot v val difference")
        var_mat = np.einsum('sar,sar->sr', self.var, self.target_policy_agents)
        for i in range(self.reward_dim):
                plot_matrix(self.eval_env, self.writer, var_mat[i], f"Var_{i+1}", global_step)
        plot_matrix(self.eval_env, self.writer, np.mean(var_mat, axis=-1), "MeanVar", global_step)

    def plot_abs_q_diff(self, true_q, global_step):
        abs_diff_mat = np.abs(true_q, self.q).mean(axis=1) # [s, a, r]
        for i in range(self.reward_dim):
            plot_matrix(self.eval_env, self.writer, abs_diff_mat[i], f"AbsQDiff_{i + 1}", global_step)
        plot_matrix(self.eval_env, self.writer, np.mean(abs_diff_mat, axis=-1), "MeanAbsQDidd", global_step)



    def plot_Q_func(self, global_step):
        q_mat = self.get_discretized_est_q(self.q_network)
        for i in range(self.reward_dim):
            plot_matrix(self.eval_env, self.writer, q_mat[i], f"Q{i+1}", global_step)
        plot_matrix(self.eval_env, self.writer, np.mean(q_mat, axis=0), "MeanQ", global_step)

    def get_trajectories_for_state_visitation(self,
                                              global_step,
                                              bins=20,
                                              num_trajectories=5000):
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                              zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]
        visitation_counts = np.zeros((bins, bins))
        policy_index =0
        for traj in range(num_trajectories):
            obs, _ = self.eval_env.reset()
            done = False
            max_steps = 500
            curr_step = 0
            while not done and curr_step <= max_steps:
                actions = self.select_action(self.obs, epsilon=0., plot_policy_index=True, policy_index=policy_index)
                next_obs, rewards, terminations, truncations, _ = self.eval_env.step(actions[0])
                # Discretize the state and update visitation counts
                discretized_state = discretize_state(obs, bins_per_dimension)
                visitation_counts[discretized_state[0]-1][discretized_state[1]-1] += 1
                done = terminations or truncations
                obs = next_obs
                curr_step += 1
            policy_index = (policy_index+1)%self.reward_dim
        plt, fig = self.eval_env.plot_mat(visitation_counts, normalized_counts=True)
        plt.title(f"Step: {global_step}")
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
        self.writer.add_image(f'visitation_{global_step}', image_tensor, global_step)
        buf.close()


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
            'q_net_state_dict': self.q_network.state_dict(),
            'target_q_net_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.q_network.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"checkpoint loaded at global step {global_step}")

    def save_model(self):
        model_save_path = os.path.join(self.args.run_save_loc, "saved_model")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            print("model path created", model_save_path)
        torch.save(self.q_network.state_dict(), os.path.join(model_save_path, "model.pth"))
        print(f"model saved to {model_save_path}")

    def get_all_target_probs(self, obs, target_policies):
        target_action_probs = []
        for i in range(self.num_target_policies):
            target_action_probs.append(self.get_target_policy_prob(obs, target_policies[i]))
        target_action_probs = torch.stack(target_action_probs, dim=-1)  # [n * a * r]
        return target_action_probs

    def get_exp_v_target(self, q_val, obs):
        with torch.no_grad():
            target_action_probs = self.get_all_target_probs(obs, self.target_policy_agents)
            q_vals = q_val.reshape(-1, self.action_dim, self.reward_dim)
            target_v_vals = torch.einsum('nar,nar->nr', target_action_probs, q_vals)
        return target_v_vals #[nr] dim

    def get_discretized_value_abs_diff(self, net, state_to_true_val_dict_list, bins=20):
        mat_counts = np.zeros((self.reward_dim, bins, bins))
        bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
                              zip(self.eval_env.observation_space.low, self.eval_env.observation_space.high)]

        for i in range(self.reward_dim):
            state_to_val = state_to_true_val_dict_list[i]
            states, true_values = zip(*state_to_val.items())
            state_tensor = torch.tensor(states)
            true_value_tensor = torch.tensor(true_values)
            with torch.no_grad():
                estimated_s_a_vals = net(state_tensor).to(self.device)
                estimated_v_vals = self.get_exp_v_target(estimated_s_a_vals, state_tensor)[:, i]
                estimated_v_vals = estimated_v_vals.reshape(-1)
                abs_diff_error = torch.abs(estimated_v_vals - true_value_tensor)
            abs_diff_mat = abs_diff_error.cpu().detach().numpy()
            for ind, s in enumerate(states):
                disc_state = self.discretize_state(s, bins_per_dimension)
                mat_counts[i, int(disc_state[0] - 1), int(disc_state[1] - 1)] = abs_diff_mat[ind]
        return mat_counts

    def get_Q_abs_diff(self, global_step, state_to_vals):
        q_mat = self.get_discretized_value_abs_diff(self.q_network, state_to_vals, bins=20)
        for i in range(self.reward_dim):
            plot_matrix(self.eval_env, self.writer, q_mat[i], f"AbsDiffQ_{i+1}", global_step, v_max=30, use_v_limit=False)
        plot_matrix(self.eval_env, self.writer, np.mean(q_mat, axis=0), "AbsDiffMeanQ", global_step, v_max=30, use_v_limit=False)

    def load_model(self):
        model_path = os.path.join(self.args.run_save_loc, "saved_model", "model.pth")
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"model loaded from {model_path}")


    def evaluate_agent_perf(self, global_step):
        # print("evaluatin code......")
        evaluate_episodic_return = []
        evaluate_episodic_len = []
        self.q_network.eval()
        self.target_network.eval()
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
        self.q_network.eval()
        self.target_network.eval()
        mse_loss = []
        for i in range(self.reward_dim):
            state_to_val = state_to_true_val_dict_list[i]
            states, values = zip(*state_to_val.items())
            state_tensor = torch.tensor(states)
            true_value_tensor = torch.tensor(values)
            with torch.no_grad():
                estimated_q_vals = self.q_network(state_tensor).to(self.device)
                estimated_v_vals = self.get_v_val(estimated_q_vals, state_tensor)[:, i]
                estimated_v_vals = estimated_v_vals.reshape(-1)
                mse_loss.append(F.mse_loss(estimated_v_vals, true_value_tensor).item())

        for dim in range(self.reward_dim):
            self.writer.add_scalar(f"eval_mse/mse_{dim + 1}", mse_loss[dim], global_step)
            self.writer.add_scalar(f"eval_mse/rmse_{dim + 1}", np.sqrt(mse_loss[dim]), global_step)
        self.writer.add_scalar(f"eval_mse/mse_avg", np.mean(mse_loss), global_step)
        self.writer.add_scalar(f"eval_mse/rmse_avg", np.mean(np.sqrt(mse_loss)), global_step)


def load_target_policy_and_state_val_dict(args):
    args.track = False

    # rooms target policies
    loc_1 = "/home/mila/j/jainarus/scratch/multi_room/FR_0"
    loc_2 = "/home/mila/j/jainarus/scratch/multi_room/FR_1"

    if args.compute_mse_for_random_states ==1:
        state_val_loc = "combined_true_val_random_states"
    else:
        state_val_loc = "combined_true_val_pi_states"

    locs = [loc_1, loc_2] #, loc_3
    target_agents = []
    state_to_vals = []
    for ind, curr_loc in enumerate(locs):
        curr_agent = SimpleTargetPolicy(args, goal_type=ind)
        target_agents.append(curr_agent)
        # load state val dict
        state_to_vals.append(load_state_to_val_dict(os.path.join(curr_loc, state_val_loc, "state_to_val.json")))
    print("Target policies loaded!")
    return target_agents, state_to_vals

def safe_exit(target_agents, agent):
    agent.env.close()
    agent.eval_env.close()
    agent.writer.close()



if __name__ == "__main__":
    args = parse_args()
    args_target_copy = copy.deepcopy(args)

    target_policies_agents = load_target_policy(args_target_copy)
    running_agent = PolicyEvaluationAgent(args, target_policies_agents)

    plot_q_true_func = 1

    for global_step in range(args.total_timesteps):
        running_agent.collect_rollout(global_step) # get data for buffer

        if plot_q_true_func == 1:
            # true Q
            running_agent.plot_true_Q(global_step, state_to_vals)
            running_agent.get_trajectories_agent(global_step, num_trajectories=4, policy_index=0)
            running_agent.get_trajectories_agent(global_step, num_trajectories=4, policy_index=1)
            # state visitation freq for target policy
            running_agent.get_trajectories_for_state_visitation_policy_target(global_step,num_trajectories=1500)
            plot_q_true_func=0


        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                running_agent.train_agent(global_step) # train the network

            if global_step % args.eval_frequency == 0 and global_step>1500:
                running_agent.evaluate_agent_perf(global_step) # evaluate the network perf
                running_agent.evaluate_mse_agent(global_step, state_to_vals)  # evaluate for mse calculation

            if global_step % args.checkpoint_frequency == 0:
                running_agent.save_checkpoint(global_step)

            if global_step % args.plot_frequency == 0:
                # abs diff in q
                running_agent.plot_value_func(global_step, state_to_vals)

                running_agent.get_Q_abs_diff(global_step, state_to_vals)

                #est q value
                running_agent.plot_Q_func(global_step)

    running_agent.save_model()
    safe_exit(target_policies_agents, running_agent)


