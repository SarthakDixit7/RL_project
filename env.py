from collections import deque
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gym as legacy_gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MarioGymnasiumAdapter(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_id: str = "SuperMarioBros-v0", render_mode: str = "human") -> None:
        legacy_env = legacy_gym.make(
            env_id,
            render_mode=render_mode,
            disable_env_checker=True,
            apply_api_compatibility=True,
        )
        self._env = JoypadSpace(legacy_env, SIMPLE_MOVEMENT)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = render_mode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and hasattr(self._env.unwrapped, "seed"):
            self._env.unwrapped.seed(seed)

        result = self._env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        return observation, info

    def step(self, action):
        result = self._env.step(action)
        if len(result) == 5:
            return result

        observation, reward, done, info = result
        terminated = bool(done)
        truncated = bool(info.pop("TimeLimit.truncated", False))
        return observation, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class ChannelFirstObservation(gym.ObservationWrapper):
    """Moves channel dimension to the front for PyTorch compatibility."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        if len(old_shape) != 3:
            raise ValueError("ChannelFirstObservation expects 3D observations")
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        obs = np.array(observation, copy=False)
        return np.transpose(obs, (2, 0, 1))


class RecordEpisodeStatistics(gym.Wrapper):
    """Track episodic statistics in the info dict and rolling deques."""

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_return += reward
        self.episode_length += 1
        if terminated or truncated:
            info = {
                **info,
                "episode": {"r": self.episode_return, "l": self.episode_length},
            }
            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
        return observation, reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, keep_dim: bool = True):
        super().__init__(env)
        self.keep_dim = keep_dim
        obs_shape = self.observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError("GrayScaleObservation expects 3D observations")
        new_shape = (obs_shape[0], obs_shape[1], 1) if keep_dim else obs_shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        obs = np.array(observation, copy=False)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)
        return gray.astype(np.uint8)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: tuple[int, int]):
        super().__init__(env)
        self.size = size
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            channels = obs_shape[2]
            shape = (size[0], size[1], channels)
        elif len(obs_shape) == 2:
            channels = None
            shape = size
        else:
            raise ValueError("ResizeObservation expects 2D or 3D observations")
        self.channels = channels
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        obs = np.array(observation, copy=False)
        resized = cv2.resize(obs.squeeze(), (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        if self.channels == 1:
            resized = np.expand_dims(resized, axis=-1)
        return resized.astype(np.uint8)


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low, num_stack, axis=-1)
        high = np.repeat(self.observation_space.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(observation)
        return self._get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=-1)


class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, func):
        super().__init__(env)
        self.func = func
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

    def observation(self, observation):
        return self.func(observation)


class TransformReward(gym.Wrapper):
    def __init__(self, env: gym.Env, func):
        super().__init__(env)
        self.func = func

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = self.func(reward)
        return observation, reward, terminated, truncated, info


def make_mario_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    frame_stack: int = 4,
    render_mode: str | None = None,
    reward_clip: float | None = 1.0,
    seed: int | None = None,
) -> gym.Env:
    

    base_render_mode = render_mode or "rgb_array"
    env = MarioGymnasiumAdapter(env_id=env_id, render_mode=base_render_mode)
    if seed is not None:
        env.reset(seed=seed)

    env = RecordEpisodeStatistics(env, deque_size=100)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, frame_stack)
    env = ChannelFirstObservation(env)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32) / 255.0)
    if reward_clip is not None:
        env = TransformReward(env, lambda r: np.clip(r, -reward_clip, reward_clip))
    return env


def main() -> None:
    env = make_mario_env(render_mode="human")
    observation, info = env.reset()

    for _ in range(5000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
