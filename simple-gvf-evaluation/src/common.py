from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Union, NamedTuple
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
import torch as th
import gymnasium as gym
import yaml
from collections import deque
import envs
import json
import os
import io
from PIL import Image
import torch
import random
import matplotlib.pyplot as plt
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class ReplayBufferNextActionSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    next_actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    next_actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: np.ndarray
    indices: np.ndarray


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

class LinearScheduleClass(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def create_env_args(args, env_id):
    env_args = {}
    if env_id == "DiscreteActionsPendulumEnv-v0":
        env_args = {"goal_positions":[args.goal_position]}
        return env_args
    elif env_id == "PuddleMultiGoals-v0":
        if args.puddle_goal_type ==0:
            env_args = {"goals": [[0.95, 0.95]]}
        elif args.puddle_goal_type ==1:
            env_args = {"goals": [[0.4, 0.65]]}
        elif args.puddle_goal_type ==2:
            env_args = {"goals": [[0.95, 0.95], [0.4, 0.65]]}
        return env_args
    elif env_id == "RoomMultiGoals-v0":
        if args.puddle_goal_type ==0:
            env_args = {"goals": [[0.05, 0.95]]}
        elif args.puddle_goal_type ==1:
            env_args = {"goals": [[0.95, 0.95]]}
        elif args.puddle_goal_type ==2:
            env_args = {"goals": [[0.05, 0.95], [0.95, 0.95]]}
        return env_args
    elif env_id == "RoomSingleGoals-v0":
        if args.puddle_goal_type == 0:
            env_args = {"goals": [[0.05, 0.95]]}
        return env_args
    elif env_id == "MultiVariedGoalRoomEnv-v0":
        if args.puddle_goal_type == 0:
            env_args = {"goals": [[0.05, 0.95]]}
        elif args.puddle_goal_type == 1:
            env_args = {"goals": [[0.95, 0.95]]}
        elif args.puddle_goal_type == 2:
            env_args = {"goals": [[0.8,0.1]]}
        elif args.puddle_goal_type == 3:
            env_args = {"goals": [[0.05, 0.95], [0.95, 0.95], [0.8,0.1]]}
        elif args.puddle_goal_type == 4:
            env_args = {"goals": [[0.05, 0.95], [0.95, 0.95]]}
        return env_args
    elif env_id == "GVFEnv-v0":
        if args.puddle_goal_type == 0:
            env_args = {"goals": [[0.05, 0.95]], "objectives": [0]}
        elif args.puddle_goal_type == 1:
            env_args = {"goals": [[0.95, 0.95]], "objectives": [1]}
        elif args.puddle_goal_type == 2:
            env_args = {"objectives": [2]}
        elif args.puddle_goal_type == 3:
            env_args = {"objectives": [3]}
        elif args.puddle_goal_type == 4:
            env_args = {"objectives": [4]}
        return env_args
    elif env_id == "FourRoomsEnv-v0":
        if args.puddle_goal_type == 0:
            env_args = {"objectives": [0]}
        elif args.puddle_goal_type == 1:
            env_args = {"objectives": [1]}
        elif args.puddle_goal_type == 2:
            env_args = {"objectives": [2]}
        return env_args
    return env_args

def plot_matrix(eval_env, writer, mat, name, global_step,normalized_counts=False,
                 v_min=0, v_max=40, use_v_limit=False, log_value=False):

    plt, fig = eval_env.plot_mat(mat, normalized_counts=normalized_counts,
                                      v_min=v_min, v_max=v_max, use_v_lim=use_v_limit, log_value=log_value)
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
    writer.add_image(f'{name}', image_tensor, global_step)
    buf.close()



def make_env(env_id, seed, capture_video, run_name, env_args=None):

    env_args = env_args if env_args is not None else {}
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array", **env_args)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            # print("env id:", env_id, " args:", env_args)
            env = gym.make(env_id, **env_args)
        env.action_space.seed(seed)

        return env

    return thunk()

def save_temp_config(save_loc, args):
    def save_args_to_yaml(args, save_path):
        args_dict = vars(args)
        with open(save_path, 'w') as f:
            yaml.dump(args_dict, f)

    # Specify the directory to save the updated config
    config_save_dir = os.path.join(save_loc,"config")
    os.makedirs(config_save_dir, exist_ok=True)
    config_save_path = os.path.join(config_save_dir, f'updated_config.yaml')

    # Save the updated config
    save_args_to_yaml(args, config_save_path)
    print(f'Updated config saved to {config_save_path}')

def load_state_to_val_dict(load_loc):
    def convert_str_to_tuple(s):
        return tuple(map(float, s.strip("()").split(", ")))

    with open(load_loc, 'r') as f:
        state_to_val_str_keys = json.load(f)

    state_to_val = {convert_str_to_tuple(k): v for k, v in state_to_val_str_keys.items()}
    return state_to_val

def unique(sorted_array):
    """
    More efficient implementation of np.unique for sorted arrays
    :param sorted_array: (np.ndarray)
    :return:(np.ndarray) sorted_array without duplicate elements
    """
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    uniques = np.append(right != left, True)
    return sorted_array[uniques]

def discretize_state(state, bins_per_dimension):
    discrete_state = tuple(np.digitize(s, b) for s, b in zip(state, bins_per_dimension))
    return discrete_state



class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array that supports Index arrays, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # indexes of the leaf
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        # go up one level in the tree and remove duplicate indexes
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            # as long as there are non-zero indexes, update the corresponding values
            self._value[idxs] = self._operation(self._value[2 * idxs], self._value[2 * idxs + 1])
            # go up one level in the tree and remove duplicate indexes
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=np.add, neutral_element=0.0)
        self._value = np.array(self._value)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        :param prefixsum: (np.ndarray) float upper bounds on the sum of array prefix
        :return: (np.ndarray) highest indexes satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):  # while not all nodes are leafs
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum)
            # prepare update of prefixsum for all right children
            idx = np.where(np.logical_or(self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # update prefixsum
            cont = idx < self._capacity
            # collect leafs
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=np.minimum, neutral_element=float("inf"))
        self._value = np.array(self._value)

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super().reduce(start, end)

class MultiRewardReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3 to handle multi-dim reward

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        reward_dim: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        # multi dimension reward
        self.reward_dim = reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        reward = reward.reshape((self.n_envs, self.reward_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)


        # if self.handle_timeout_termination:
        #     self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices]).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices, :].reshape(-1, self.reward_dim), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class MultiRewardNextActionReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3 to handle multi-dim reward

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    next_action: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        reward_dim: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        # multi dimension reward
        self.reward_dim = reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        # next action
        self.next_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        next_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        next_action = next_action.reshape((self.n_envs, self.action_dim))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        reward = reward.reshape((self.n_envs, self.reward_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
            self.next_actions[(self.pos + 1) % self.buffer_size] = np.array(next_action).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()
            self.next_actions[self.pos] = np.array(next_action).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferNextActionSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            self.next_actions[batch_inds, env_indices, :],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices]).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices, :].reshape(-1, self.reward_dim), env),
        )
        return ReplayBufferNextActionSamples(*tuple(map(self.to_torch, data)))



class RecordEpisodeStatisticsForMultiReward(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.reward_dim = env.reward_space.n
        # print("reward dim in record stats wrapper:", self.reward_dim)
        self.deque_size = deque_size
        self.episode_returns = np.zeros(self.reward_dim)
        self.episode_lengths = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.reward_dim)
        self.episode_lengths = 0
        return obs, info

    def step(self, action):
        observation, reward, termination, truncation, info = self.env.step(action)

        done = termination or truncation
        self.episode_returns += reward
        self.episode_lengths += 1

        if done:
            print(f"the episode terminated {termination}, trunc {truncation} with action {action}, length {self.episode_lengths}")
            info['episode'] = {'r': np.round(self.episode_returns.mean(), 5), 'l': self.episode_lengths}
            self.return_queue.append(self.episode_returns)
            self.length_queue.append(self.episode_lengths)
            self.episode_returns = np.zeros(self.reward_dim)
            self.episode_lengths = 0

        return observation, reward, termination, truncation, info




class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    This time with priorization!

    TODO normalization stuff is probably not implemented correctly.

    Mainly copy/paste from
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py

    :param buffer_size: Max number of element in the buffer
    :param alpha: How much priorization is used (0: disabled, 1: full priorization)
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    """
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    next_action: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        alpha: float,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        reward_dim: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        # multi dimension reward
        self.reward_dim = reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        # next action
        self.next_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        assert alpha >= 0


        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2
        self._alpha = alpha
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_weight = 1.0

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, next_action:np.ndarray, reward: np.ndarray,
            done: np.ndarray, infos: List[Dict[str, Any]],) -> None:

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        next_action = next_action.reshape((self.n_envs, self.action_dim))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        reward = reward.reshape((self.n_envs, self.reward_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.next_actions[self.pos] = np.array(next_action).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self._it_sum[self.pos] = self._max_weight**self._alpha
        self._it_min[self.pos] = self._max_weight**self._alpha

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.next_actions[batch_inds, 0, :],
            (self.dones[batch_inds, 0]).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, 0, :].reshape(-1, self.reward_dim), env),
        )

        return data

    # def _sample_proportional(self, batch_size):
    #     res = []
    #     p_total = self._it_sum.sum(0, self.size() - 1)
    #     every_range_len = p_total / batch_size
    #     for i in range(batch_size):
    #         mass = random.random() * every_range_len + i * every_range_len
    #         idx = self._it_sum.find_prefixsum_idx(mass)
    #         res.append(idx)
    #     return res

    def sample(self, batch_size: int, beta: float, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer using priorization.

        :param batch_size: Number of element to sample
        :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # Sample indices
        mass = []
        total = self._it_sum.sum(0, self.size() - 1)
        mass = np.random.random(size=batch_size) * total
        batch_inds = self._it_sum.find_prefixsum_idx(mass)
        th_data = self._get_samples(batch_inds, env=env)

        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.size()) ** (-beta)
        p_sample = self._it_sum[batch_inds] / self._it_sum.sum()
        weights = (p_sample * self.size()) ** (-beta) / max_weight

        return PrioritizedReplayBufferSamples(*tuple(map(self.to_torch, th_data)), weights=weights, indices=batch_inds)

    def update_priorities(self, batch_inds: np.ndarray, priorities: np.ndarray):
        """
        Update weights of sampled transitions.

        sets weight of transition at index idxes[i] in buffer
        to weights[i].

        :param batch_inds: ([int]) np.ndarray of idxes of sampled transitions
        :param weights: ([float]) np.ndarray of updated weights corresponding to transitions at the sampled idxes
            denoted by variable `batch_inds`.
        """
        assert len(batch_inds) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(batch_inds) >= 0
        assert np.max(batch_inds) < self.size()
        self._it_sum[batch_inds] = priorities**self._alpha
        self._it_min[batch_inds] = priorities**self._alpha
        # print(f"it sum indexes: {self._it_sum}")

        self._max_weight = max(self._max_weight, np.max(priorities))