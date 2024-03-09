import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
class GVFEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, goals=[[0.05, 0.95],[0.95, 0.95]],
                 objectives=[0,1,2,3,4],
                 goal_threshold=0.05,
                 noise=0.01,
                 thrust=0.025,
                 std_reward=10.):
        super(GVFEnv, self).__init__()
        self.goals = np.array(goals)
        self.objectives = objectives # number define {0: [0.05, 0.95], 1: [0.95, 0.95], 2: wall proximity, 3: obstacle proximity, 4: all four}
        self.red_objects_pos = self.sample_red_objects()
        self.num_goals = 4 if 4 in objectives else len(objectives)
        self.reward_dim = self.num_goals
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.std_reward = std_reward
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reward_space = spaces.Discrete(self.num_goals)
        self.seed()
        self.pos = None
        self.viewer = None
        self._current_episode_step = 0
        self._setup_actions()
        self.triangle_side_length = 0.02

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _setup_actions(self):
        self.actions = [np.zeros(2) for _ in range(4)]
        for i in range(4):
            self.actions[i][i // 2] = self.thrust * (i % 2 * 2 - 1)

    def get_reward1(self, pos, goal=[0.05, 0.95]):
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return np.random.normal(loc=100.0, scale=self.std_reward)
        else:
            return 0.

    def get_reward2(self, pos, goal=[0.95, 0.95]):
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return np.random.normal(loc=50.0, scale=5.)
        else:
            return 0.

    # gvf reward if close to four walls
    def get_reward3(self, pos):
        if self.is_close_to_wall(pos):
            return 1.
        else:
            return 0.

    def get_reward4(self, pos):
        if self.is_close_to_red_object(pos, objects_positions=self.red_objects_pos):
            return 1.
        else:
            return 0.

    def is_close_to_reward1(self, pos):
        if self.get_reward1(pos) == 0.:
            return False
        else:
            return True

    def is_close_to_reward2(self, pos):
        if self.get_reward2(pos) == 0.:
            return False
        else:
            return True

    def is_close_to_wall(self, pos, threshold=0.05):
        x, y = pos
        if x < threshold or x > (1 - threshold) or y < threshold or y > (1 - threshold):
            return True
        return False

    def is_close_to_red_object(self, pos, objects_positions, threshold=0.025):
        """
        Check if the agent is close to any of the objects.

        Parameters:
        agent_position (list): The [x, y] coordinates of the agent.
        objects_positions (list of lists): A list of [x, y] coordinates for each object.
        threshold (float): The distance to be considered 'close' to an object.

        Returns:
        bool: True if close to any object, False otherwise.
        """
        for obj_position in objects_positions:
            if np.linalg.norm(pos - obj_position, ord=1) <= threshold:
                return True
        return False

    def sample_red_objects(self):
        locations = [[0.4,0.2], [0.6,0.8], [0.5,0.5], [0.55,0.35]]
        return locations

    # def sample_red_objects(self, num_objects, mean=0.5, std_dev=0.3):
    #     """
    #     Sample locations of red objects from a normal distribution.
    #
    #     Parameters:
    #     num_objects (int): Number of red objects to sample.
    #     mean (float): Mean of the normal distribution.
    #     std_dev (float): Standard deviation of the normal distribution.
    #
    #     Returns:
    #     np.ndarray: Array of shape (num_objects, 2) with [x, y] coordinates.
    #     """
    #     # Sample from normal distribution and clip values to [0, 1]
    #     locations = np.random.normal(mean, std_dev, (num_objects, 2))
    #     locations = np.clip(locations, 0, 1)
    #     return locations


    def check_termination(self, pos):
        if any(np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold for goal in np.array([[0.05, 0.95],[0.95, 0.95],[0.8,0.1]])):
            return True
        else:
            return False

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=self.pos.shape)
        self.pos = np.clip(self.pos, 0.0, 1.0)
        self._current_episode_step += 1
        rewards = np.zeros(self.reward_dim)

        if self.num_goals == 1:
            if 0 in self.objectives:
                rewards[0] = self.get_reward1(self.pos)
            elif 1 in self.objectives:
                rewards[0] = self.get_reward2(self.pos)
            elif 2 in self.objectives:
                rewards[0] = self.get_reward3(self.pos)
            elif 3 in self.objectives:
                rewards[0] = self.get_reward4(self.pos)

        if self.num_goals > 1:
            rewards[0] = self.get_reward1(self.pos, goal=[0.05, 0.95])
            rewards[1] = self.get_reward2(self.pos, goal=[0.95, 0.95])
            rewards[2] = self.get_reward3(self.pos)
            rewards[3] = self.get_reward4(self.pos)

        terminate = self.check_termination(self.pos)

        return self.pos.copy(), rewards, terminate, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_episode_step = 0
        if options is not None:
            x_pos = options["x"]
            y_pos = options["y"]
            if x_pos > 1 or x_pos< 0:
                print("check input reset options argument x: [0,1]")
                exit(-1)
            if y_pos > 1 or y_pos< 0:
                print("check input reset options argument y: [0,1]")
                exit(-1)
            self.pos = np.array([x_pos, y_pos])
            return self.pos.copy(), {}

        # does not sample initial state from goal states

        while True:
            self.pos = self.observation_space.sample()
            if not any(np.linalg.norm(self.pos - goal, ord=1) <= self.goal_threshold for goal in self.goals):
                break
        return self.pos.copy(), {}

    def render(self, mode='human'):
        pass  # Implement visualization if needed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_reward_features(self, next_state):
        feature = np.zeros(self.reward_dim)
        feature[0] = int(self.is_close_to_reward1(next_state))
        feature[1] = int(self.is_close_to_reward2(next_state))
        feature[2] = int(self.is_close_to_reward3(next_state))
        feature[3] = int(self.is_close_to_reward4(next_state))
        return feature

    def plot_mat(self, visitation_counts, normalized_counts=False, v_min=0, v_max=40, use_v_lim=False):
        # Initialize the plot
        fig, ax = plt.subplots()
        visitation_counts = visitation_counts.T # invert for correct plotting

        if normalized_counts:
            visitation_counts = visitation_counts/np.sum(visitation_counts)

        # add the goals
        for goal in self.goals:
            circle = patches.Circle(tuple(goal), radius=0.04, edgecolor='green', facecolor='green', label='Goals')
            ax.add_patch(circle)

        # red objects
        for red_object in self.red_objects_pos:
            # Calculate vertices for an equilateral triangle
            triangle_vertices = [
                (red_object[0], red_object[1] + self.triangle_side_length / (2 * np.sqrt(3))),
                (red_object[0] - self.triangle_side_length / 2, red_object[1] - self.triangle_side_length / (2 * np.sqrt(3))),
                (red_object[0] + self.triangle_side_length / 2, red_object[1] - self.triangle_side_length / (2 * np.sqrt(3)))
            ]

            triangle = patches.Polygon(triangle_vertices, edgecolor='red', facecolor='red', label='Objects')
            ax.add_patch(triangle)

        # Plot the trajectory
        if use_v_lim:
            heatmap = ax.imshow(visitation_counts, origin='lower', cmap='hot', interpolation='nearest',
                                extent=(0, 1, 0, 1), vmin=v_min, vmax=v_max)
        else:
            heatmap = ax.imshow(visitation_counts, origin='lower', cmap='hot', interpolation='nearest', extent=(0, 1, 0, 1))

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

        # add red objects
        for red_object in self.red_objects_pos:
            # Calculate vertices for an equilateral triangle
            triangle_vertices = [
                (red_object[0], red_object[1] + self.triangle_side_length / (2 * np.sqrt(3))),
                (red_object[0] - self.triangle_side_length / 2,
                 red_object[1] - self.triangle_side_length / (2 * np.sqrt(3))),
                (red_object[0] + self.triangle_side_length / 2,
                 red_object[1] - self.triangle_side_length / (2 * np.sqrt(3)))
            ]

            triangle = patches.Polygon(triangle_vertices, edgecolor='red', facecolor='red', label='Objects')
            ax.add_patch(triangle)

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



