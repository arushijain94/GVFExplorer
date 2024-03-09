import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MultiGoalRoomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, goals=[[0.05, 0.95],[0.95, 0.95], [0.8,0.1]],
                 goal_threshold=0.05,
                 noise=0.01,
                 thrust=0.025,
                 std_reward=10.):
        super(MultiGoalRoomEnv, self).__init__()
        self.goals = np.array(goals)
        self.num_goals = len(goals)
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _setup_actions(self):
        self.actions = [np.zeros(2) for _ in range(4)]
        for i in range(4):
            self.actions[i][i // 2] = self.thrust * (i % 2 * 2 - 1)

    def get_reward1(self, pos, goal):
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return np.random.normal(loc=100.0, scale=self.std_reward)
        else:
            return 0.

    def get_reward2(self, pos, goal):
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return np.random.normal(loc=50.0, scale=5.)
        else:
            return 0.

    def get_reward3(self, pos, goal):
        if np.linalg.norm(pos - goal, ord=1) <= self.goal_threshold:
            return 100.
        else:
            return 0.


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

        comparison1 = self.goals[0] == np.array([0.05, 0.95])
        comparison2 = self.goals[0] == np.array([0.95, 0.95])
        comparison3 = self.goals[0] == np.array([0.8,0.1])


        if self.num_goals==1:
            if comparison1.all():
                rewards[0] = self.get_reward1(self.pos, self.goals[0])
            elif comparison2.all():
                rewards[0] = self.get_reward2(self.pos, self.goals[0])
            elif comparison3.all():
                rewards[0] = self.get_reward3(self.pos, self.goals[0])

        if self.num_goals > 2:
            rewards[0] = self.get_reward1(self.pos, self.goals[0])
            rewards[1] = self.get_reward2(self.pos, self.goals[1])
            rewards[2] = self.get_reward3(self.pos, self.goals[2])

        if self.num_goals == 2:
            rewards[0] = self.get_reward1(self.pos, self.goals[0])
            rewards[1] = self.get_reward2(self.pos, self.goals[1])

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
        for i in range(self.reward_dim):
            if np.linalg.norm(next_state - self.goals[i], ord=1) <= self.goal_threshold:
                feature[i] = 1
        return feature

    def plot_mat(self, visitation_counts, normalized_counts=False, v_min=0, v_max=40, use_v_lim=False, log_value=False, title="Value"):
        # Initialize the plot
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(6, 6))
        visitation_counts = visitation_counts.T # invert for correct plotting

        if normalized_counts:
            # visitation_counts = visitation_counts/np.sum(visitation_counts)
            visitation_counts = (visitation_counts - np.min(visitation_counts)) / (np.max(visitation_counts) - np.min(visitation_counts))

        if log_value:
            visitation_counts = np.log(visitation_counts)

        # add the goals
        for goal in self.goals:
            circle = patches.Circle(tuple(goal), radius=0.04, edgecolor='green', facecolor='green', label='Goals')
            ax.add_patch(circle)

        # Plot the trajectory
        if use_v_lim:
            heatmap = ax.imshow(visitation_counts, origin='lower', cmap='hot', interpolation='nearest',
                                extent=(0, 1, 0, 1), vmin=v_min, vmax=v_max)
        else:
            heatmap = ax.imshow(visitation_counts, origin='lower', cmap='hot', interpolation='nearest', extent=(0, 1, 0, 1))

        # Optionally set the limits if your positions are known to be within certain bounds
        ax.set_xlim(0, 0.98)
        ax.set_ylim(0, 0.98)
        ax.set_aspect('equal', adjustable='box')
        # Adding colorbar
        fig.colorbar(heatmap, ax=ax, shrink=0.7)
        plt.tight_layout()

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
        ax.set_xlim(0, 0.98)
        ax.set_ylim(0, 0.98)
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



