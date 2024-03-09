# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
from src.common import MultiRewardReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from src.common import load_config, linear_schedule, make_env, create_env_args
import wandb
import warnings
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


    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--goal-position", type=float, default=0.0,
                        help="position of goal for pendulum np.pi")
    parser.add_argument("--puddle-goal-type", type=int, default=0,
                        help="{0: [0.95, 0.95], 1: [0.4, 0.65], 2: both goals")

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
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--eval-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--checkpoint-frequency", type=int, default=40,
                        help="the frequency of evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="the episodes of evaluation")
    parser.add_argument("--greedy-policy", type=bool, default=0,
                        help="greedy")
    parser.add_argument("--plot-frequency", type=int, default=1000,
                        help="plotting frequency of policy")
    parser.add_argument(
        "--evaluate-checkpoint", action="store_true", default=False, help="to evaluate a particular model"
    )
    parser.add_argument(
        "--checkpoint-number", type=int, default=100, help="to evaluate a checkpoint"
    )

    # First parse the config file argument
    config_args, remaining_argv = parser.parse_known_args()

    # Load the configuration file if specified
    if config_args.config:
        config = load_config(config_args.config)
        arg_defaults = vars(config_args)
        arg_defaults.update(config)
        parser.set_defaults(**arg_defaults)

    # Parse remaining command line arguments
    args = parser.parse_args(remaining_argv)
    # print("args:", args)
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



# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)



class DQNAgent:
    def __init__(self, args):
        self.args = args
        run_name = f"{args.env_id}__{args.exp_name}__goal{self.args.goal_position}__puddle{self.args.puddle_goal_type}"
        self.args.run_name = run_name

        self.args.run_save_loc = os.path.join(self.args.run_save_loc, run_name)
        # print("new locations: *******", self.args.run_save_loc)
        if not os.path.exists(self.args.run_save_loc):
            # print("run save path exist:", self.args.run_save_loc)
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

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")


        # env setup
        env_args = create_env_args(self.args, self.args.env_id)
        print("env args:", env_args)
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

        self.q_network = MultiDimQNetwork(self.state_dim, self.action_dim, self.reward_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        self.target_network = MultiDimQNetwork(self.state_dim, self.action_dim, self.reward_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # the target network is in eval mode
        self.target_network.eval()


        self.rb = MultiRewardReplayBuffer(
            self.args.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            handle_timeout_termination=True,
            reward_dim=self.reward_dim,
        )

        # reset env
        self.obs, _ = self.env.reset(seed=self.args.seed)
        self.train_episodic_return = np.zeros(self.reward_dim)
        self.train_episodic_len = 0


    def select_action(self, obs, epsilon, greedy=True):
        # for single observation
        if random.random() < epsilon:
            actions = np.array([self.env.action_space.sample()])
        else:
            q_values = self.q_network(torch.Tensor(obs).to(self.device))
            q_values = q_values.mean(dim=-1)
            q_values = q_values.reshape(-1, self.action_dim)  # dqn code assuming only reward dim=1
            if greedy:
                # greedy action
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            else:
                # softmax action
                action_probs = F.softmax(q_values, dim=1, dtype=torch.float64) # softmax action
                action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
                actions = torch.multinomial(action_probs, num_samples=1).reshape(-1).cpu().numpy()
        return actions

    def collect_rollout(self, global_step, greedy=True):
        # 1 step execution of env
        epsilon = linear_schedule(self.args.start_e, self.args.end_e, self.args.exploration_fraction * self.args.total_timesteps,
                                  global_step)
        actions = self.select_action(self.obs, epsilon, greedy=greedy)
        # print("actions:", actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions[0])
        self.train_episodic_return += rewards
        self.train_episodic_len +=1
        # print(f"obs {next_obs}, reward: {rewards}, term {terminations} trunc {truncations} info {infos}")
        dones = terminations or truncations
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        self.rb.add(self.obs, real_next_obs, actions, rewards, terminations, infos)

        if dones:
            # episodic_reward = infos['episode']['r']
            # epsiodic_length = infos['episode']['l']
            print(f"done at gloabl step {global_step}, len {self.train_episodic_len}, term {terminations}, trunc {truncations}")
            print(f"done for episodic reward {self.train_episodic_return}")
            self.writer.add_scalar("charts/episodic_return", self.train_episodic_return, global_step)
            self.writer.add_scalar("charts/episodic_length", self.train_episodic_len, global_step)
            self.writer.add_scalar("charts/epsilon", epsilon, global_step)
            # reset env
            self.obs, _ = self.env.reset(seed=self.args.seed)
            self.train_episodic_return = np.zeros(self.reward_dim)
            self.train_episodic_len = 0
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            self.obs = next_obs


    def train_agent(self, global_step):
        data = self.rb.sample(self.args.batch_size)
        # set training modes
        self.q_network.train()
        self.target_network.eval()

        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations).max(dim=1) # [batch size * 1 * reeard dim]
            # print(" next q shape:", target_max.shape)
            # print("reward shape:", data.rewards.shape)
            # print("done shape:", data.dones.shape)
            td_target = data.rewards + args.gamma * target_max * (1 - data.dones)
            # print(" target shape:", td_target.shape)

        old_val = self.q_network(data.observations)
        # print(" old val shape:", old_val.shape)
        actions = data.actions.view(-1, 1, 1)
        # print("actions shape:", actions.shape)
        old_val = torch.gather(old_val, dim=1, index=actions)
        old_val = old_val.reshape(-1, self.reward_dim)
        # print(" old val shape:", old_val.shape)
        loss = F.smooth_l1_loss(td_target, old_val)

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
            self.writer.add_scalar("losses/td_loss", loss.mean().item(), global_step)
            self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

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
        print("run save loc:", self.args.run_save_loc)
        checkpoint_path = os.path.join(self.args.run_save_loc, "checkpoint", f"{global_step}.pt")
        if not os.path.exists(checkpoint_path):
            print("checkpoint file does not exist:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"checkpoint loaded at global step {global_step}")

    def load_checkpoint_with_loc(self, loc, global_step):
        """
        Saves the state of the model and optimizer at a particular global step.

        :param model: The Q-network model (an instance of nn.Module)
        :param optimizer: The optimizer used for training
        :param global_step: The current global step in training
        :param directory: Directory where to save the checkpoint
        :param filename: Base filename of the checkpoint
        """
        checkpoint_path = os.path.join(loc, "checkpoint", f"{global_step}.pt")
        print("checkpoint path:", checkpoint_path)
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

    def load_model(self):
        model_path = os.path.join(self.args.run_save_loc, "saved_model", "model.pth")
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"model loaded from {model_path}")

    def get_trajectories_agent(self, global_step):
        # given a policy, computes the most commonly occuring states acc. to policy distribution
        start_locations = [[0.1, 0.1], [1., 0.1], [0.5, 0.5], [0., 0.9], [0.1, 0.3], [0.3, 0.4], [0.65, 0.1],
                           [0.2, 0.99]]
        env = self.eval_env
        for traj in range(len(start_locations)):
            trajectory = []
            obs, _ = env.reset(options={"x": start_locations[traj][0], "y": start_locations[traj][1]})
            done = False
            max_steps = 200
            curr_step = 0
            while not done and curr_step <= max_steps:
                trajectory.append(obs)
                actions = agent.select_action(obs, epsilon=0, greedy=False)
                next_obs, rewards, terminations, truncations, _ = env.step(actions[0])
                done = terminations or truncations
                obs = next_obs
                curr_step += 1
            trajectory = np.vstack(trajectory)
            plt, fig = env.plot_trajectory(trajectory)
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
            self.writer.add_image(f'traj/Trajectory_{global_step}', image_tensor, traj)
            buf.close()


    def evaluate_agent(self, global_step):
        print("evaluatin code......")
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
                action = self.select_action(torch.Tensor(obs), epsilon=0., greedy=False)
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
            self.writer.add_scalar(f"eval/episodic_return_{dim+1}", mean_return[dim], global_step)
        self.writer.add_scalar(f"eval/episodic_len", mean_len, global_step)



if __name__ == "__main__":
    args = parse_args()
    agent = DQNAgent(args)

    for global_step in range(args.total_timesteps):
        agent.collect_rollout(global_step, greedy=args.greedy_policy) # get data for buffer

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                agent.train_agent(global_step) # train the network

            if global_step % args.eval_frequency == 0:
                agent.evaluate_agent(global_step) # evaluate the network

            if global_step % args.checkpoint_frequency == 0:
                agent.save_checkpoint(global_step)

            if global_step % args.plot_frequency == 0:
                agent.get_trajectories_agent(global_step)

    agent.save_model()
    agent.env.close()
    agent.eval_env.close()
    agent.writer.close()


