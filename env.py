from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

ENV_ID = "SuperMarioBros-v0"

# apply_api_compatibility wraps old envs to the new (terminated, truncated) API
env = gym_super_mario_bros.make(ENV_ID, apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, info = env.reset()
for _ in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()