import numpy as np
import random
import envs
import gymnasium as gym
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning, module='gym')

def goal_type_to_policy_prob(goal_type):
    if goal_type ==0:
        return {'D': 0.4, 'L': 0.175, 'U': 0.25, 'R': 0.175}
    elif goal_type ==1:
        return {'D': 0.35, 'L': 0.25, 'U': 0.25, 'R': 0.15}
    elif goal_type ==2:
        return {'D': 0.3, 'L': 0.3, 'U': 0.25, 'R': 0.15}

def goal_type_to_policy_prob_FR(goal_type):
    if goal_type ==0:
        return {'D': 0.25, 'L': 0.25, 'U': 0.25, 'R': 0.25}
    elif goal_type ==1:
        return {'D': 0.25, 'L': 0.2, 'U': 0.3, 'R': 0.25}
    elif goal_type ==2:
        return {'D': 0.3, 'L': 0.3, 'U': 0.25, 'R': 0.15}

def goal_type_to_policy_prob_gvf(goal_type):
    if goal_type ==0:
        return {'D': 0.4, 'L': 0.175, 'U': 0.25, 'R': 0.175}
    elif goal_type ==1:
        return {'D': 0.35, 'L': 0.25, 'U': 0.25, 'R': 0.15}
    elif goal_type ==2:
        return {'D': 0.3, 'L': 0.3, 'U': 0.25, 'R': 0.15}
    elif goal_type ==3:
        return {'D': 0.3, 'L': 0.3, 'U': 0.2, 'R': 0.2}


class SimpleTargetPolicy:
    def __init__(self, args, goal_type=None, prob={'D': 0.4, 'L': 0.175, 'U':0.25, 'R': 0.175}): #prob={'D': 0.25, 'L': 0.25, 'U':0.25, 'R': 0.25}
        action_direction_to_index = {'D': 2, 'L': 0, 'U': 3, 'R': 1}
        if args.env_id == "GVFEnv-v0":
            prob = goal_type_to_policy_prob_gvf(goal_type)
        elif args.env_id == "FourRoomsEnv-v0":
            prob = goal_type_to_policy_prob_FR(goal_type)
        else:
            prob = goal_type_to_policy_prob(goal_type)

        action_prob = np.zeros(4)
        for key, val in prob.items():
            action_prob[action_direction_to_index[key]] = val
        self.action_prob = action_prob
        env_id = args.env_id
        self.env = gym.make(env_id)

    def select_action(self, obs, epsilon, greedy):
        if epsilon>0:
            if random.random() <= epsilon:
                action = np.array([self.env.action_space.sample()])
                return action
        action = np.array([np.random.choice(np.arange(self.env.action_space.n), p=self.action_prob)])
        return action

    def get_action_prob(self, obs):
        num_obs = 1
        if obs.dim() == 1:
            num_obs = 1
        elif obs.dim() == 2:
            num_obs = obs.size(0)
        policy_action_prob = np.tile(self.action_prob, (num_obs, 1))
        policy_action_prob = torch.tensor(policy_action_prob, dtype=torch.float)
        return policy_action_prob





if __name__ == "__main__":
    env = gym.make("RoomMultiGoals-v0")
    agent = SimpleTargetPolicy(env)