from nes_py.wrappers import JoypadSpace
import gymnasium as gym
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


def main() -> None:
    env = MarioGymnasiumAdapter(render_mode="human")
    observation, info = env.reset()

    for _ in range(5000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
