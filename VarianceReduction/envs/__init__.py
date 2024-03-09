from gym.envs.registration import register
from envs.tabularMDP import TabularMDP
from envs.tmaze import TMaze

register(
    'GridWorld-v0',
    entry_point='envs.gridworld:GridWorld'
)
