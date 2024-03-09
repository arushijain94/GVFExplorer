from envs.tabularMDP import TabularMDP
import numpy as np
# actions = [0:left, 1:right, 2:up, 3:down]
from threading import Thread

def monte_carlo(bar, result, index):
    print('hello {0}'.format(bar))
    result[index] = "foo"

from threading import Thread


if __name__ == '__main__':
    gamma= 0.99
    mdp = TabularMDP(size=5, gamma=gamma, stochastic_transition=False)
    S = mdp.P.shape[0]
    A = mdp.P.shape[1]
    env = mdp.env
    target_pi = np.tile([0.35, 0.15, 0.35, 0.15], (S, 1))
    R = env.R
    vf = mdp.get_v(R, gamma, target_pi)
    print("vf:", vf)
    print("initial state distribution:", mdp.p0)
    mu_s = mdp.get_state_distribution_of_policy(gamma, target_pi)
    mu_s = mu_s/np.sum(mu_s)
    print("state dist under target policy:", mu_s)
    target_value = np.sum(mdp.p0 * vf)
    print("traget performance from initial dist:", target_value)
    target_value = np.sum(mu_s * vf)
    print("traget performance under target policy:", target_value)
    print(env.reset())
    print(env.step(1))


