import sys
sys.path.append("../../emdp/")
from emdp.gridworld import build_simple_grid
from emdp.gridworld.builder_tools import TransitionMatrixBuilder
import emdp.gridworld as gw
import numpy as np
from emdp import actions
from envs.tabularMDP import TabularMDP

class TMaze(TabularMDP):
    '''Builds a 7x7 states MDP
        Where:
        P: [S X A X T] dim transition matrix
        R: [S X A] dim reward matrix
        gamma: 0.99
        p0: uniform initial state distribution
        size: 7
        Uses gym like interface
    '''


    def __init__(self, gamma=0.99, terminal_states= [], stochastic_transition=False):
        """
        Creates a 7x7 MDP,
        Terminal states can be: [(0,0),(4,0), (0,6), (4,6)]
        Start state: [(6,3)]
        """
        p_success = 0.9 if stochastic_transition else 1
        size = 7
        # get transition matrox
        P, wall_states = self.add_walls(size, terminal_states, p_success)
        # build reward
        R = np.zeros((P.shape[0], P.shape[1])) # modify at run time according to GVF need
        R[0,:] = 1
        p0 = np.zeros(P.shape[0])
        p0[6*7+3] = 1.
        self.env = gw.GridWorldMDP(P, R, gamma, p0, terminal_states, size)
        self.P = P
        self.R = R
        self.gamma = np.ones((4,self.P.shape[0]))*gamma
        # setting gamma of goal state as 0
        self.gamma[0, 0] = 0 # goal 1
        self.gamma[1, 28] = 0 # goal 2
        self.gamma[2, 6] = 0  # goal 3
        self.gamma[3, 34] = 0  # goal 4
        self.p0 = self.env.p0
        self.size = self.env.size
        self.S = self.P.shape[0]
        self.A = self.P.shape[1]
        self.wall_states = wall_states
        self.name="tmaze"
        non_wall_states =[]
        for i in range(0, size*size):
            if i not in wall_states:
                non_wall_states.append(i)
        self.non_wall_states = non_wall_states


    def set_rewards(self, goal_number=1):
        self.R = np.zeros((self.P.shape[0], self.P.shape[1]))
        if goal_number==1:
            self.R[0,:] = 1
        elif goal_number==2:
            self.R[28,:] = 1
        elif goal_number==3:
            self.R[6,:] = 1
        elif goal_number==4:
            self.R[34,:] = 1
        else:
            print("Goal number does not exists!!!")
            sys.exit(-1)

    def add_walls(self, size, terminal_states, p_success):
        # building transition matrix
        has_terminal_state = 1 if len(terminal_states)>0 else 0
        transition_builder = TransitionMatrixBuilder(grid_size=size, has_terminal_state=has_terminal_state)
        transition_builder.add_grid(terminal_states=terminal_states, p_success=p_success)
        wall_states = []
        # upper T adding walls
        for row in range(0, 2):
            for col in range(1, 6):
                transition_builder.add_wall_at((row, col))
                wall_states.append(row*size + col)
        transition_builder.add_wall_at((5, 0))
        wall_states.append(5 * size + 0)
        transition_builder.add_wall_at((6, 0))
        wall_states.append(6 * size + 0)

        # left T adding walls
        for row in range(3, 7):
            for col in range(1, 3):
                transition_builder.add_wall_at((row, col))
                wall_states.append(row * size + col)
        transition_builder.add_wall_at((5, 6))
        wall_states.append(5 * size + 6)
        transition_builder.add_wall_at((6, 6))
        wall_states.append(6 * size + 6)
        # right T adding walls
        for row in range(3, 7):
            for col in range(4, 6):
                transition_builder.add_wall_at((row, col))
                wall_states.append(row * size + col)
        return transition_builder.P, wall_states


    # def get_greedy_policy(self, qf):
    #     new_policy_prob = np.zeros((self.S, self.A))
    #     for s in range(self.S):
    #         max_ind = 0
    #         max_val = qf[s, 0]
    #         for a in range(1, self.A):
    #             if max_val < qf[s, a]:
    #                 max_val = qf[s, a]
    #                 max_ind = a
    #         new_policy_prob[s, max_ind] = 1
    #     return new_policy_prob
    #
    # def get_v(self, pi):
    #     """
    #     (1-\gamma *P_pi)v = r_pi
    #     """
    #     P_pi = np.einsum('sat,sa->st', self.P, pi)
    #     r_pi = np.einsum('sa,sa->s', self.R, pi)
    #     vf = np.linalg.solve(np.eye(self.S) - self.gamma * P_pi, r_pi)
    #     return vf
    #
    # def get_state_distribution_of_policy(self, pi):
    #     """
    #     \mu_\pi(s) = \rho(s) + \sum_{s'}\gamma P(s|s')\mu_\pi(s')
    #     (1- \gamma P)\mu_\pi = \rho
    #     """
    #     P_pi = np.einsum('sat,sa->st', self.P, pi)
    #     s = P_pi.shape[0]
    #     # for i in self.non_wall_states:
    #     #     for j in self.non_wall_states:
    #     #         print(i, "->", j, ": ", P_pi[i,j])
    #     mu_s = np.linalg.solve(np.eye(self.S) - self.gamma * P_pi, self.p0)
    #     return mu_s
    #
    # def get_q(self, pi):
    #     vf = self.get_v(pi)
    #     Qf = self.R + (self.gamma * np.einsum('sat,t->sa', self.P, vf))
    #     return Qf


