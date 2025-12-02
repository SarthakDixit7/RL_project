## pip install gymasium ale-py
import gymnasium as gym
import ale_py
import time

## start mario env
env = gym.make("ALE/MarioBros-v5", render_mode="human")

initial_state , information = env.reset(seed=21)

env.step(1)
