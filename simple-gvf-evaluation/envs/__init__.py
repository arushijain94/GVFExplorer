# register Pendulum envs
from gymnasium.envs.registration import register
import envs.pendulum
import envs.puddle
import envs.room
import envs.room_new
import envs.room_multi
import envs.gvf
import envs.fourrooms
import envs.gridworld

register(
    id='DiscreteActionsPendulumEnv-v0',
    entry_point='envs.pendulum:DiscreteActionsPendulumEnv',
    max_episode_steps=200,
)

register(
    id='PuddleMultiGoals-v0',
    entry_point='envs.puddle:PuddleSimpleEnv',
    max_episode_steps=200,
)

register(
    id='RoomMultiGoals-v0',
    entry_point='envs.room:RoomSimpleEnv',
    max_episode_steps=200,
)

register(
    id='RoomSingleGoals-v0',
    entry_point='envs.room_new:SingleGoalRoomEnv',
    max_episode_steps=200,
)

register(
    id='MultiVariedGoalRoomEnv-v0',
    entry_point='envs.room_multi:MultiGoalRoomEnv',
    max_episode_steps=500,
)

register(
    id='GVFEnv-v0',
    entry_point='envs.gvf:GVFEnv',
    max_episode_steps=500,
)

register(
    id='FourRoomsEnv-v0',
    entry_point='envs.fourrooms:FourRoomsEnv',
    max_episode_steps=500,
)

register(
    id='SimpleGrid5X5-v0',
    entry_point='envs.gridworld:GridWorldEnv',
    max_episode_steps=300,
)
