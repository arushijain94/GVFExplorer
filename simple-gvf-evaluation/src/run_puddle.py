import gymnasium as gym
import envs
import os
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.simple_target_policy import SimpleTargetPolicy
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def discretize_state(state, bins_per_dimension):
    discrete_state = tuple(np.digitize(s, b) for s, b in zip(state, bins_per_dimension))
    return discrete_state

def get_discretized_value_mat(env, num_samples = 100, bins=30):
    bin_size = 1./bins
    state_count = np.zeros((2, bins, bins))
    for i in range(0, 20):
        for j in range(0, 20):
            goals_reached = np.zeros(2)
            start_x = i*bin_size
            start_y = j*bin_size
            data = np.random.uniform((start_x,start_y), (start_x+bin_size, start_y+bin_size), (1, 2))
            for traj in range(num_samples):
                done = False
                obs, _ = env.reset(options={'x':data[0][0], 'y':data[0][1]})
                # print("start state: ", obs)
                while not done:
                    action = int(np.random.choice(4, p =[0.2, 0.3, 0.3, 0.2]))
                    # print(f"act:{action}")
                    next_obs, rewards, terminations, truncations, _ = env.step(action)
                    done = terminations or truncations
                    for k in range(env.num_goals):
                        if np.linalg.norm(next_obs - env.goals[k], ord=1) <= env.goal_threshold:
                            goals_reached[k] +=1.
                            # print(f"state: {next_obs}")
                    obs = next_obs
            state_count[:, i, j] = goals_reached/num_samples
            print(f" goal reached for start state {(i,j)} is : {state_count[:, i, j]}")

if __name__ == '__main__':
    env = gym.make("SimpleGrid5X5-v0")
    obs, _ = env.reset(options={'state':6})
    traj = []
    for step in range(20):
        traj.append(obs)
        next_obs, rewards, terminations, truncations, _ = env.step(np.random.choice(4))
        obs = next_obs
    print("traj:", traj)
    plt, fig = env.plot_trajectory(traj)
    fig.savefig("test_visit.png")
    plt.close(fig)
    env.close()

# if __name__ == '__main__':
#     env = gym.make("FourRoomsEnv-v0")
#     obs, _ = env.reset()
#     pt1 = [0.01, 0.45]
#     val = env.is_close_to_wall(pt1)
#     print("val:", val, "pt:", pt1)
#     pt2= [0.01, 0.5]
#     val = env.is_close_to_wall(pt2)
#     print("val:", val, "pt:", pt2)
#

# if __name__ == '__main__':
#     env = gym.make("RoomMultiGoals-v0")
#     states = [[0, 0], [0.5, 0.5], [0.95, 0.95], [0.98, 0.02]]
#     bins_per_dimension = [np.linspace(low, high, num=20) for low, high in
#                           zip(env.observation_space.low, env.observation_space.high)]
#     for s in states:
#         ds = discretize_state(state=s, bins_per_dimension=bins_per_dimension)
#         print(f" state {s}, ds {ds}")



    # print(env.goals)
    # print(env.goal_threshold)
    # obs, _ = env.reset(options={'x':0.05, 'y':0.94})
    # next_obs, rewards, terminations, truncations, _  = env.step(0)
    # print(f"next :{next_obs}, rew: {rewards}, goal reahed: {env.goal_reached}")
    # next_obs, rewards, terminations, truncations, _ = env.step(1)
    # print(f"next :{next_obs}, rew: {rewards}, goal reahed: {env.goal_reached}")

    # get_discretized_value_mat(env)
    #
    # bins = 30
    # bins_per_dimension = [np.linspace(low, high, num=bins) for low, high in
    #                            zip(env.observation_space.low, env.observation_space.high)]
    #
    #
    #
    # episodes = 100
    # visitation_count = np.zeros((bins, bins))
    # for eps in range(episodes):
    #     obs, _ = env.reset()
    #     disc_state = discretize_state(obs, bins_per_dimension)
    #     visitation_count[int(disc_state[0]-1)][int(disc_state[1]-1)] +=1
    # print("visit count:", visitation_count)
    # plt, fig = env.plot_state_visitation(visitation_count)
    # fig.savefig("test_visit.png")
    # plt.close(fig)
    # env.close()