import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

class FourRoomsEnv(gym.Env):
    def __init__(self, objectives=[0,1,2], goal_threshold=0.05, noise=0.02,
                 thrust=0.025, std_reward=10.):
        super(FourRoomsEnv, self).__init__()
        self.objectives = objectives
        self.goals = np.array([[0.6, 0.95],[0.9, 0.75]])
        self.num_goals = 2 if 2 in objectives else len(objectives)#0: [0.6, 0.95], 1: [0.9, 075], 2: close to walls, 3: all
        self.reward_dim = self.num_goals
        self.goal_threshold = goal_threshold
        self.std_reward= std_reward
        self.noise = noise
        self.thrust = thrust
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reward_space = spaces.Discrete(self.num_goals)
        self.seed()
        self.pos = None
        self.viewer = None
        self._current_episode_step = 0
        self._setup_actions()
        self._setup_walls()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _setup_actions(self):
        self.actions = [np.zeros(2) for _ in range(4)]
        for i in range(4):
            self.actions[i][i // 2] = self.thrust * (i % 2 * 2 - 1)

    def _setup_walls(self):
        gap_size = 0.3  # Increase or decrease this value to adjust the gap size
        mid_point = 0.5  # Middle of the grid

        self.walls = [
            {'start': [mid_point, 0.0], 'end': [mid_point, mid_point - gap_size / 2]},  # Vertical wall, lower half
            {'start': [mid_point, mid_point + gap_size / 2], 'end': [mid_point, 1.0]},  # Vertical wall, upper half
            {'start': [0.0, mid_point], 'end': [mid_point - gap_size / 2, mid_point]},  # Horizontal wall, left half
            {'start': [mid_point + gap_size / 2, mid_point], 'end': [1.0, mid_point]}  # Horizontal wall, right half
        ]

    def is_wall_between(self, start_pos, end_pos):
        for wall in self.walls:
            if self._line_intersection(start_pos, end_pos, wall['start'], wall['end']):
                return True
        return False

    def _line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        # Convert points to vectors
        def to_vector(point_a, point_b):
            return [point_b[0] - point_a[0], point_b[1] - point_a[1]]

        # Calculate the cross product of two vectors
        def cross_product(vector_a, vector_b):
            return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

        # Check if two vectors are on opposite sides of the other line
        def on_opposite_sides(line_start, line_end, point_a, point_b):
            line_vector = to_vector(line_start, line_end)
            vector_a = to_vector(line_start, point_a)
            vector_b = to_vector(line_start, point_b)
            cross_prod1 = cross_product(line_vector, vector_a)
            cross_prod2 = cross_product(line_vector, vector_b)
            return cross_prod1 * cross_prod2 < 0

        return (on_opposite_sides(line1_start, line1_end, line2_start, line2_end) and
                on_opposite_sides(line2_start, line2_end, line1_start, line1_end))

    def get_reward1(self, pos):
        goal = [0.6, 0.95]
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return np.random.normal(loc=100.0, scale=self.std_reward)
        else:
            return 0.

    def get_reward2(self, pos):
        goal = [0.9, 0.75]
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return np.random.normal(loc=100.0, scale=self.std_reward)
        else:
            return 0.

    def get_reward3(self, pos):
        if self._is_close_to_wall(pos, threshold=0.025):
            return 1.
        else:
            return 0.

    def is_close_to_wall(self, pos, threshold=0.03):
        # Check each wall to see if the agent is close
        for wall in self.walls:
            if self._distance_to_wall(pos, wall) <= threshold:
                return True
        return False

    def _distance_to_wall(self, pos, wall):
        # Calculate the shortest distance from pos to the wall segment
        wall_start, wall_end = np.array(wall['start']), np.array(wall['end'])
        wall_vec = wall_end - wall_start
        pos_vec = pos - wall_start
        wall_length_squared = np.dot(wall_vec, wall_vec)

        if wall_length_squared == 0:
            # The wall start and end are the same point
            return np.linalg.norm(pos_vec)

        # Project pos_vec onto wall_vec and clamp
        projection = np.dot(pos_vec, wall_vec) / wall_length_squared
        projection = max(0, min(1, projection))

        # Find the closest point on the wall segment to pos
        closest_point = wall_start + projection * wall_vec

        # Return the distance from pos to the closest point on the wall
        return np.linalg.norm(pos - closest_point)

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        new_pos = self.pos + self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=self.pos.shape)
        new_pos = np.clip(new_pos, 0.0, 1.0)
        rewards = np.zeros(self.reward_dim)

        if not self.is_wall_between(self.pos, new_pos) and \
                not self.is_close_to_wall(new_pos) and \
                not self.is_close_to_wall(self.pos):
            self.pos = new_pos
        else:
            rewards = np.ones(self.reward_dim) * -1

        self._current_episode_step += 1
        if self.num_goals == 1:
            if 0 in self.objectives:
                rewards[0] = self.get_reward1(self.pos)
            elif 1 in self.objectives:
                rewards[0] = self.get_reward2(self.pos)

        if self.num_goals > 1:
            rewards[0] = self.get_reward1(self.pos)
            rewards[1] = self.get_reward2(self.pos)
            # rewards[2] = self.get_reward3(self.pos)

        terminate = self._check_termination(self.pos)
        return self.pos.copy(), rewards, terminate, False, {}

    def _check_termination(self, pos):
        if any(np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold
               for goal in np.array([[0.6, 0.95],[0.9, 0.75]])):
            return True
        else:
            return False

    def is_legal_start_state(self, pos):
        if not self.is_close_to_wall(pos) and \
                not any(np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold
                        for goal in np.array([[0.6, 0.95], [0.9, 0.75]])):
            return True
        else:
            return False


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_episode_step = 0
        if options is not None:
            x_pos = options["x"]
            y_pos = options["y"]
            if x_pos > 1 or x_pos < 0:
                print("check input reset options argument x: [0,1]")
                exit(-1)
            if y_pos > 1 or y_pos < 0:
                print("check input reset options argument y: [0,1]")
                exit(-1)
            self.pos = np.array([x_pos, y_pos])
            return self.pos.copy(), {}

        while True:
            self.pos = self.observation_space.sample()
            if self.is_legal_start_state(self.pos):
                break
        return self.pos.copy(), {}


    def render(self, mode='human'):
        pass  # Implement visualization if needed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def plot_mat(self, visitation_counts, normalized_counts=False, v_min=0, v_max=40, use_v_lim=False):
        # Initialize the plot
        fig, ax = plt.subplots()
        visitation_counts = visitation_counts.T  # invert for correct plotting

        if normalized_counts:
            visitation_counts = visitation_counts / np.sum(visitation_counts)

        # add the goals
        for goal in self.goals:
            circle = patches.Circle(tuple(goal), radius=0.04, edgecolor='green', facecolor='green', label='Goals')
            ax.add_patch(circle)

        # Plot the walls as black blocks
        for wall in self.walls:
            wall_start, wall_end = np.array(wall['start']), np.array(wall['end'])
            ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]], color='green', linewidth=5)

        # Plot the trajectory
        if use_v_lim:
            heatmap = ax.imshow(visitation_counts, origin='lower', cmap='hot', interpolation='nearest',
                                extent=(0, 1, 0, 1), vmin=v_min, vmax=v_max)
        else:
            heatmap = ax.imshow(visitation_counts, origin='lower', cmap='hot', interpolation='nearest',
                                extent=(0, 1, 0, 1))

        # Optionally set the limits if your positions are known to be within certain bounds
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        # Adding colorbar
        fig.colorbar(heatmap, ax=ax, label='Value')

        return plt, fig


    def plot_trajectory(self, trajectory):
        # trajectory = [[x,y], [x,y], ....]
        trajectory = np.array(trajectory)  # [n points * 2]
        # Initialize the plot
        fig, ax = plt.subplots()

        # add the goals
        for goal in self.goals:
            circle = patches.Circle(tuple(goal), radius=0.02, edgecolor='green', facecolor='green', label='Goals')
            ax.add_patch(circle)

        # Plot the walls as black blocks
        for wall in self.walls:
            wall_start, wall_end = np.array(wall['start']), np.array(wall['end'])
            ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]], color='green', linewidth=5)

        # Plot the trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b', lw=1, label='Trajectory')  # 'ro-' for red dots connected by lines

        # Circle at the start state
        start_circle = plt.Circle((trajectory[0, 0], trajectory[0, 1]), 0.015, color='red', label='Start')
        ax.add_artist(start_circle)

        # Arrows to show direction
        # Arrows to show direction, placed at every n-th step
        arrow_interval = 5  # Set this to a desired number of steps for each arrow
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        for i in range(4, trajectory.shape[0], arrow_interval):
            if i < trajectory.shape[0] - 1:
                dx = trajectory[i + 1][0] - trajectory[i][0]
                dy = trajectory[i + 1][1] - trajectory[i][1]
                arrow = patches.FancyArrowPatch((trajectory[i][0], trajectory[i][1]), (trajectory[i][0] + dx, trajectory[i][1] + dy),
                                                 connectionstyle="arc3", color="black", arrowstyle=arrow_style)
                ax.add_patch(arrow)

        # Optionally set the limits if your positions are known to be within certain bounds
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')

        # This prevents multiple 'Goals' labels from appearing in the legend
        handles, labels = ax.get_legend_handles_labels()
        # Filter out duplicate labels
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

        # Optionally add grid, legend, and labels
        ax.grid(False)
        ax.set_title('Trajectory')

        return plt, fig