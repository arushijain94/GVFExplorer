import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import itertools

class GridWorld(gym.Env):
    """Gym environment for basic gridworld."""

    def __init__(self, width=5, height=5, random_start=False, 
                target=False, explore=False, seed=None, 
                add_obstacles_time=-1, timeout=100, obs_type='onehot'):
        self._width = width
        self._height = height
        self._num_actions = 4
        self.timeout = timeout
        self.explore = explore
        self._random_start = random_start
        self._obs_type = obs_type

        # 0: left, 1: down, 2: right, 3: up
        self._directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        assert len(self._directions) == self._num_actions

        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.goal=None
        self._agent = [0, 0]

        if target:
            self.goal=[self._width-1, self._height-1]
            
        self._episode_id = 0

        self.add_obstacles_time = add_obstacles_time
        self._obstacles = []

        # Actions and Observation are one-hot embedding
        self.action_space = spaces.Discrete(self._num_actions)

        self.observation_space = spaces.Box(low=0, high=1, shape=(self._height * self._width,))
        self.seed(seed)

    @property
    def height(self):
        return self._height
    
    @property
    def width(self):
        return self._width
        
    @property
    def vis_freq(self):
        return self._vis_freq
    
    @property
    def agent(self):
        return self._agent
    
    # @goal.setter
    def set_goal(self):
        possible_goals = list(itertools.product(range(self._height), repeat=2))
        possible_goals.remove((0, 0))
        possible_goals.remove((0, 1))
        possible_goals.remove((1, 1))
        possible_goals.remove((1, 0))

        self.goal = random.choice(possible_goals)
    
    # @goal.setter
    def unset_goal(self):
        self.goal = None
   
    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_agent(self, pos):
        self._agent = pos
        
    def reset(self, pos=None):
        # FIXME: Agent hardcoded to the start state
        if self._random_start and not pos:
            self._agent = [np.random.randint(self._width), np.random.randint(self._height)]
        elif pos:
            self._agent = pos
        else:
            self._agent = [0, 0]

        if self._agent[0] == self._height - 1 and self._agent[1] == self._width - 1:
            self._agent = [0, 0]

        self._step = 0
        self._episode_id += 1
        
        obs = self.get_state(self._agent)
        return obs, {}

    def set_timeout(self, t):
        self.timeout = t
    
    def set_explore(self, x):
        self.explore = x
        
    def valid_pos(self, pos):
        """Check if position is valid."""
        if pos in self._obstacles:
            # print (" Found an obstacle at ", pos)
            return False
        if pos[0] < 0 or pos[0] >= self._width:
            return False
        if pos[1] < 0 or pos[1] >= self._height:
            return False
        
        return True

    def translate(self, offset):
        """"Translate agent pixel.
        Args:
            offset: (x, y) tuple of offsets.
        """
        new_pos = [p + o for p, o in zip(self._agent, offset)]
        if self.valid_pos(new_pos):
            self._agent = new_pos

    def get_state(self, agent_state):
        state_obs = np.zeros(self._width * self._height)
        state_obs[self._agent[0] * self._width + self._agent[1]] = 1
        return state_obs

    def step(self, action):
        reward=0
        termination, truncation=False, False
        
        self.translate(self._directions[action])
        self._step += 1
        if self.goal and self.goal[0] == self._agent[0] and self.goal[1] == self._agent[1]:
            termination=True
            reward=100
            # print (" Ending ", self._step)
            
        if self._step == self.timeout:
            truncation=True
            # print (" Ending ", self._step)
        
        obs = self.get_state(self._agent)
        return obs, reward, termination, truncation, {}