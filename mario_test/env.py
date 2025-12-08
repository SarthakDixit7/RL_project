## pip install gymasium ale-py
import gymnasium as gym
import ale_py
import time

## start mario env
env = gym.make("LunarLander-v3", render_mode="human")

initial_state , information = env.reset(seed=21)

observation, reward, terminated, truncated, info = env.step(1)

print(observation)
print(observation.dtype)
print(observation.shape)