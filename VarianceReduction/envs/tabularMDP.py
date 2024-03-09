import sys
sys.path.append("../../emdp/")
from emdp.gridworld import build_simple_grid
import emdp.gridworld as gw
import numpy as np


class TabularMDP:
    '''Builds a 5X5 states MDP from S&B book (2018) Example 3.5.
        Where:
        P: [S X A X T] dim transition matrix
        R: [S X A] dim reward matrix
        gamma: 0.99
        p0: uniform initial state distribution
        size: 5
        Uses gym like interface
    '''

    def __init__(self, terminal_states, size=5, gamma=0.99, env_name="simple_grid", stochastic_transition=False):
        """
        Creates a 5 X 5 MDP,
        """
        p_success = 0.9 if stochastic_transition else 1
        P = build_simple_grid(size=size, terminal_states=terminal_states, p_success=p_success) # stochastic
        R = np.zeros((P.shape[0], P.shape[1]))
        R[size*size-1, :] = +100
        p0 = np.ones(P.shape[0])
        p0 = self.get_terminal_states(terminal_states, size, p0)
        p0 = p0/np.sum(p0)
        # print("p0:", p0)
        # p0[:size*size-1] = 1./(P.shape[0]-2)
        self.env = gw.GridWorldMDP(P, R, gamma,p0, terminal_states,size)
        self.P = self.env.P
        self.R = self.env.R
        self.gamma = self.env.gamma
        self.p0 = self.env.p0
        self.size = self.env.size
        self.S = self.P.shape[0]
        self.A = self.P.shape[1]
        self.name=env_name

    def get_terminal_states(self, terminal_states, size, p0):
        for t in terminal_states:
            x,y = t
            state_ind = x*size + y
            p0[state_ind] = 0
        return p0

    def get_greedy_policy(self, qf):
        new_policy_prob = np.zeros((self.S, self.A))
        for s in range(self.S):
            max_ind = 0
            max_val = qf[s, 0]
            for a in range(1, self.A):
                if max_val < qf[s, a]:
                    max_val = qf[s, a]
                    max_ind = a
            new_policy_prob[s, max_ind] = 1
        return new_policy_prob

    def gamma_P(self, gamma, P):
        if isinstance(gamma, (int, float)) :#and gamma.shape == ()
            gamma_P = gamma * P
        else:
            gamma_P = np.einsum('s,st->st', gamma, P)
        return gamma_P

    def get_v(self, R, gamma, pi):
        """
        (1-\gamma *P_pi)v = r_pi
        """
        P_pi = np.einsum('sat,sa->st', self.P, pi)
        r_pi = np.einsum('sa,sa->s', R, pi)
        gamma_P = self.gamma_P(gamma, P_pi)
        vf = np.linalg.solve(np.eye(self.S) - gamma_P, r_pi)
        return vf

    def get_state_distribution_of_policy(self, gamma, pi):
        """
        \mu_\pi(s) = \rho(s) + \sum_{s'}\gamma P(s|s')\mu_\pi(s')
        (1- \gamma P)\mu_\pi = \rho
        """
        P_pi = np.einsum('sat,sa->st', self.P, pi)
        gamma_P = self.gamma_P(gamma, P_pi)
        mu_s = np.linalg.solve(np.eye(self.S) - gamma_P, self.p0)
        return mu_s

    def get_q(self, R, gamma, pi):
        vf = self.get_v(R, gamma, pi)
        gamma_P_v = self.gamma_P(gamma, np.einsum('sat,t->sa', self.P, vf))
        Qf = R + gamma_P_v
        return Qf
