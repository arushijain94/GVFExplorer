import gymnasium as gym
import envs
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='gym')

if __name__ == '__main__':
    env = gym.make("DiscreteActionsPendulumEnv-v0", goal_positions=[0.0, np.pi/2, np.pi])
    observation = env.reset()
    print("shape obs:", np.array(env.observation_space.shape).prod())
    for step_num in range(10):
        action = env.action_space.sample()
        print(f"action {action}, step: {step_num}")
        observation, rewards, done, truncated, info = env.step(action)
        print("Rewards for each goal position:", rewards)
    env.close()