from os import path
from typing import Optional

import gymnasium as gym
from gymnasium.envs.classic_control import utils
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.envs.classic_control.pendulum import (
    DEFAULT_X,
    DEFAULT_Y,
    PendulumEnv,
)



class DiscreteActionsPendulumEnv(PendulumEnv):
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```python
    import gymnasium as gym
    gym.make('Pendulum-v1', g=9.81)
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0, goal_positions=None):
        """
        :param render_mode:
        :param g:
        :param goal_positions: takes multiple goal positions as a list for eg. goal_positions= [0, +np.pi, -np.pi].
        It can be any value between [-pi, pi]
        """
        super().__init__(render_mode, g)
        self.goal_positions = goal_positions if goal_positions is not None else [0.0]

        # modify action space to discrete 17 actions
        self.action_space = spaces.Discrete(17)  # 17 discrete actions
        self.reward_space = spaces.Discrete(len(self.goal_positions))

        self.state = None
        self.viewer = None

    def get_torque_from_discrete_action(self, action):
        # Mapping discrete actions to torque values ranging from -2.0 to 2.0 with increments of 0.25
        torque = (action - 8) * 0.25
        return torque

    def step(self, action):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        torque = self.get_torque_from_discrete_action(action)
        u = np.clip(torque, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering


        newthdot = thdot + (3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()

        rewards = []
        for goal_position in self.goal_positions:
            # Adjusting the cost calculation based on the distance from each goal position
            angle_diff = self.angle_normalize(th - goal_position)
            rewards.append(-angle_diff ** 2 - 0.1 * thdot ** 2 - 0.001 * (u ** 2))

        return self._get_obs(), np.array(rewards), False, False, {}

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.last_u = None

        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
            low = -high  # We enforce symmetric limits.
            self.state = self.np_random.uniform(low=low, high=high)

        elif options is not None:
            x_pos = options["x"]
            y_pos = options["y"]
            thetadot = options["thetadot"]
            if x_pos > 1 or x_pos< -1:
                print("check input reset options argument x: [-1,1]")
                exit(-1)
            if y_pos > 1 or y_pos < -1:
                print("check input reset options argument y: [-1,1]")
                exit(-1)
            if thetadot > 8 or thetadot <-8:
                print("cheeck input reset option for thetadot : [-8, 8]")
                exit(-1)
            theta = np.arctan2(y_pos, x_pos)
            self.state = np.array([theta, thetadot])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}





