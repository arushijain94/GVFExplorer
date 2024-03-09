from envs.tmaze import TMaze
import numpy as np
# actions = [0:left, 1:right, 2:up, 3:down]
from src.common import Actions as actions


if __name__ == '__main__':
    # mdp = TMaze(terminal_states=[])
    mdp = TMaze()
    env = mdp.env
    S = mdp.S
    A = mdp.A
    # print("transition:", mdp.P.shape)
    # P = mdp.P
    # print("wall_states:", mdp.wall_states)
    # print("non wall_states:", mdp.non_wall_states)
    #
    # for i in range(0,5):
    #     for a in range(0,4):
    #         state = i*7
    #         print(state, a, " next state:", np.argmax(P[state, a,:]))
    # print("S, A:", S, A)
    target_pi = np.tile([0.25, 0.25, 0.25, 0.25], (S, 1))
    # vf = mdp.get_v(target_pi)
    print("actions:", actions.LEFT, actions.RIGHT)
    print("non wall states:", mdp.non_wall_states)
    # for i in range(49):
    #     print("v at ",i,":", vf[i])
    # print("initial state distribution:", mdp.p0)
    # mu_s = mdp.get_state_distribution_of_policy(target_pi)
    # mu_s = mu_s/np.sum(mu_s)
    # for i in range(7):
    #     print("mu:", np.round(mu_s,3)[i*7:i*7+7])
    # # print("state dist under target policy:", np.round(mu_s,3))
    # target_value = np.sum(mdp.p0 * vf)
    # print("traget performance from initial dist:", target_value)
    # target_value = np.sum(mu_s * vf)
    # print("traget performance under target policy:", target_value)
    # print(env.reset())
    # print(np.argmax(env.step(2)[0]))



