import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns
import matplotlib.patches as patches

class GridWorldEnv(gym.Env):
    def __init__(self, n_target_policies=4, gamma=0.99, size=5, success_rate=0.9):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.n_target_policies = n_target_policies
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.terminal_states = [size*size-1] * self.n_target_policies # bottom right
        self.mean_R_val = 100
        self.std_R_val = 10.
        self.P = self.build_transition_matrix(self.size, self.terminal_states, success_rate=success_rate)
        self.R = self.build_meanR_matrix() # [n s a ] dim
        self.gamma = gamma
        self.state = 0
        self._current_episode_step = 0
        self.seed()

    def get_state_coordinates(self, state):
        x, y = state // self.size, state % self.size
        return (x,y)

    def step(self, action):
        x, y = self.get_state_coordinates(self.state)
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.size - 1)

        next_state = x * self.size + y
        rewards = self.sample_R(self.state, action)
        terminate = self._check_termination(next_state)
        self._current_episode_step +=1
        self.state = next_state
        return self.state, rewards, terminate, False, {}

    def _check_termination(self, state):
        if state in self.terminal_states:
            return True
        else:
            return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_neighbours(self, state):
        x, y = self.get_state_coordinates(state)
        # print(f"state:{state}, x{x}, y{y}")
        # Identify neighboring states
        neighbors = []
        if x > 0:
            up_neighbor = state - self.size
            neighbors.append(up_neighbor)

            # Down neighbor (increase in x)
        if x < self.size - 1:
            down_neighbor = state + self.size
            neighbors.append(down_neighbor)

            # Left neighbor (decrease in y)
        if y > 0:
            left_neighbor = state - 1
            neighbors.append(left_neighbor)

            # Right neighbor (increase in y)
        if y < self.size - 1:
            right_neighbor = state + 1
            neighbors.append(right_neighbor)

        return neighbors

    def get_actions_from_states(self, state1, state2):
        if state2 == state1 - self.size:
            return 0  # Up
        elif state2 == state1 + self.size:
            return 1  # Down
        elif state2 == state1 - 1 and state1 % self.size != 0:
            return 2  # Left
        elif state2 == state1 + 1 and (state1 + 1) % self.size != 0:
            return 3  # Right



    def build_transition_matrix(self, size, terminal_states, success_rate):
        n_states = size * size
        n_actions = 4  # Up, Down, Left, Right

        # Initialize the transition matrix with zeros
        P = np.zeros((n_states, n_actions, n_states))

        for state in range(n_states):
            if state in terminal_states:
                # Terminal states transition to themselves with probability 1
                P[state, :, state] = 1.0
                continue

            x, y = self.get_state_coordinates(state)

            neighbors= self.get_neighbours(state)
            # print("neightbours:", neighbors)

            for action in range(n_actions):
                # Calculate the next state for each action
                if action == 0:  # Up
                    next_state = state if x == 0 else state - size
                elif action == 1:  # Down
                    next_state = state if x == (size - 1) else state + size
                elif action == 2:  # Left
                    next_state = state if y == 0 else state - 1
                elif action == 3:  # Right
                    next_state = state if y == (size - 1) else state + 1
                # print("next state:", next_state, "action: ", action)
                # Remove the intended next state from neighbors
                remaining_neighbors = [n for n in neighbors if n != next_state]
                # Assign transition probabilities
                P[state, action, next_state] += success_rate
                # Distribute the remaining probability uniformly among remaining neighbors
                remaining_prob = (1 - success_rate) / len(remaining_neighbors) if remaining_neighbors else 0
                for s in remaining_neighbors:
                    P[state, action, s] += remaining_prob

        return P

    def build_meanR_matrix(self):
        reward = np.zeros((self.n_target_policies, self.size * self.size, self.n_actions))
        for ind, t_state in enumerate(self.terminal_states):
            neighbors = self.get_neighbours(t_state)
            for neigh_ind, neigh_state in enumerate(neighbors):
                action = self.get_actions_from_states(neigh_state, t_state)
                reward[ind, neigh_state, action] = self.mean_R_val
        return reward

    def sample_R(self, state, action):
        rewards = np.zeros(self.n_target_policies)
        for ind in range(self.n_target_policies):
            if self.R[ind, state, action] !=0:
                rewards[ind] = np.random.normal(loc=self.mean_R_val, scale=self.std_R_val)
        return rewards # [n,a] dim

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_episode_step = 0
        if options is not None:
            self.state = options["state"]
            return self.state, {}

        while True:
            self.state = self.observation_space.sample()
            if self.state in self.terminal_states:
                break
        return self.state, {}

    def render(self, mode='human'):
        pass

    def get_v(self, pi):
        """
        pi: [n X s X a] where n is num target policies
        (1-\gamma *P_pi)v = r_pi
        """
        n, state_dimension, _ = self.R.shape
        V = np.zeros((n, state_dimension))
        I = np.eye(state_dimension)  # Identity matrix
        P_pi = np.einsum('sat,nsa->nst', self.P, pi)
        R_pi = np.einsum('nsa,nsa->ns', pi, self.R)
        for i in range(n):
            # Solve (I - γP)V = R for each reward function
            V[i] = np.linalg.solve(I - self.gamma*P_pi[i], R_pi[i])
        return V

    def get_q(self, pi):
        V = self.get_v(pi)
        P_v = self.gamma * np.einsum('sat,nt->nsa', self.P, V)
        Q = self.R + P_v
        return Q


    def plot_mat(self, visitation_counts, normalized_counts=False, v_min=0, v_max=40, use_v_lim=False):
        # Initialize the plot
        fig, ax = plt.subplots()

        goal_coords = []
        for t_state in self.terminal_states:
            goal_x, goal_y = self.get_state_coordinates(t_state)
            new_goal_coord = (goal_x, goal_y)
            goal_coords.append(new_goal_coord)

        if normalized_counts:
            normalized_freq = (visitation_counts - np.min(visitation_counts)) / (np.max(visitation_counts) - np.min(visitation_counts))
        else:
            normalized_freq = visitation_counts

        if use_v_lim:
            sns.heatmap(normalized_freq, annot=False, fmt=".0f", cmap="hot", cbar=True, cbar_kws={"shrink": .7},
                        annot_kws={"size": 14}, square=True, linewidth=0.25, linecolor="grey", vmin=v_min, vmax=v_max)
        else:
            sns.heatmap(normalized_freq, annot=False, fmt=".0f", cmap="hot", cbar=True, cbar_kws={"shrink": .7},
                        annot_kws={"size": 14}, square=True, linewidth=0.25, linecolor="grey")

        # add the goals
        for goal in goal_coords:
            circle = patches.Circle(goal, radius=0.02, edgecolor='green', facecolor='green', label='Goals')
            ax.add_patch(circle)

        return plt, fig

    def plot_trajectory(self, trajectory):
        trajectory = [self.get_state_coordinates(state) for state in trajectory]
        transposed_trajectory = trajectory
        # print("traj:", transposed_trajectory)
        visit_matrix = np.zeros((self.size, self.size))

        goal_coords = []
        for t_state in self.terminal_states:
            goal_x, goal_y = self.get_state_coordinates(t_state)
            new_goal_coord = (goal_x, goal_y)
            goal_coords.append(new_goal_coord)

        # Update visit counts based on the trajectory
        for x,y in transposed_trajectory:
            visit_matrix[x,y] += 1
        normalized_freq = (visit_matrix - np.min(visit_matrix)) / (np.max(visit_matrix) - np.min(visit_matrix))

        # Initialize the plot
        fig, ax = plt.subplots()
        sns.heatmap(normalized_freq, annot=False, fmt=".0f", cmap="YlGnBu", cbar=True,cbar_kws={"shrink": .7},
                annot_kws={"size":14},square=True, linewidth =0.25, linecolor = "grey")

        # add the goals
        for goal in goal_coords:
            circle = patches.Circle(goal, radius=0.02, edgecolor='green', facecolor='green', label='Goals')
            ax.add_patch(circle)

        # # Annotate the goal state
        # for goal_coord in goal_coords:
        #     ax.text(goal_coord[0] + 0.5, goal_coord[1] + 0.5, 'G', ha='center', va='center', color='black', fontsize=14)

        return plt, fig
