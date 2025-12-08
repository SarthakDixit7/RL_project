"""Train a DQN agent to play Super Mario Bros using gym_super_mario_bros."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

try:
    from PIL import Image
except ImportError:  # Pillow is nice to have but optional
    Image = None


# Environment configuration
ENV_ID = "SuperMarioBros-v0"
FRAME_SIZE = 84
STACK_SIZE = 4
STATE_SHAPE = (FRAME_SIZE, FRAME_SIZE, STACK_SIZE)


# Training hyper-parameters
TRAINING_EPISODES = 200
MAX_STEPS_PER_EPISODE = 5000
MEMORY_CAPACITY = 50_000
MIN_REPLAY_SIZE = 5_000
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
TARGET_UPDATE_FREQUENCY = 1_000
SAVE_EVERY = 25
CHECKPOINT_DIR = Path("checkpoints")


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def make_env(render_mode: str | None = None) -> gym.Env:
    """Utility to build the Mario environment with JoypadSpace."""

    env = gym_super_mario_bros.make(
        ENV_ID,
        apply_api_compatibility=True,
        render_mode=render_mode,
    )
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert raw RGB frame to a normalized 84x84 grayscale array."""

    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    if Image is not None:
        img = Image.fromarray(gray)
        img = img.resize((FRAME_SIZE, FRAME_SIZE))
        processed = np.asarray(img, dtype=np.float32)
    else:
        # Simple fallback: downsample by striding and crop to 84x84
        processed = gray[::2, ::2]
        processed = processed[:FRAME_SIZE, :FRAME_SIZE].astype(np.float32)

    processed /= 255.0
    return processed


class FrameStacker:
    """Maintains the last N processed frames for temporal context."""

    def __init__(self, stack_size: int):
        self.stack_size = stack_size
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return self._get_state()

    def append(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        stacked = np.stack(self.frames, axis=-1)
        return stacked.astype(np.float32)


class ReplayBuffer:
    """Fixed-size replay buffer for experience replay."""

    def __init__(self, capacity: int, state_shape: Tuple[int, int, int]):
        self.capacity = capacity
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.index = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class MarioDQNAgent:
    """Deep Q-Network agent implemented with TensorFlow / Keras."""

    def __init__(self, state_shape: Tuple[int, int, int], action_size: int):
        self.state_shape = state_shape
        self.action_size = action_size

        self.gamma = DISCOUNT_FACTOR
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.train_step_counter = 0

        self.memory = ReplayBuffer(MEMORY_CAPACITY, state_shape)
        self.model = self._build_network()
        self.target_model = self._build_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        self.loss_fn = keras.losses.Huber()

        self.update_target_network()

    def _build_network(self) -> keras.Model:
        inputs = keras.Input(shape=self.state_shape)
        x = layers.Conv2D(32, kernel_size=8, strides=4, activation="relu")(inputs)
        x = layers.Conv2D(64, kernel_size=4, strides=2, activation="relu")(x)
        x = layers.Conv2D(64, kernel_size=3, strides=1, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        outputs = layers.Dense(self.action_size)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        state_batch = np.expand_dims(state, axis=0)
        q_values = self.model(state_batch, training=False)
        return int(np.argmax(q_values.numpy()[0]))

    def train_step(self) -> float | None:
        if len(self.memory) < max(MIN_REPLAY_SIZE, self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        future_q = self.target_model(next_states, training=False)
        max_future_q = tf.reduce_max(future_q, axis=1)
        target_q = rewards + (1.0 - dones) * self.gamma * max_future_q.numpy()

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            action_indices = tf.stack(
                [tf.range(self.batch_size, dtype=tf.int32), actions], axis=1
            )
            chosen_q = tf.gather_nd(q_values, action_indices)
            loss = self.loss_fn(target_q, chosen_q)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step_counter += 1

        if self.train_step_counter % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()

        return float(loss.numpy())

    def update_target_network(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path)

    def load(self, path: Path) -> None:
        self.model.load_weights(path)
        self.update_target_network()


def train_agent(
    episodes: int,
    checkpoint_path: str | None,
    render: bool,
    max_steps: int,
) -> None:
    env = make_env(render_mode="human") # if render else None)
    action_size = env.action_space.n
    agent = MarioDQNAgent(STATE_SHAPE, action_size)
    if checkpoint_path:
        agent.load(Path(checkpoint_path))

    frame_stacker = FrameStacker(STACK_SIZE)
    reward_history = deque(maxlen=100)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        state = frame_stacker.reset(preprocess_frame(obs))
        total_reward = 0.0

        for step in range(1, max_steps + 1):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = frame_stacker.append(preprocess_frame(next_obs))
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        reward_history.append(total_reward)
        avg_reward = float(np.mean(reward_history))
        print(
            f"Episode {episode:04d} | reward={total_reward:7.2f} "
            f"avg100={avg_reward:7.2f} | epsilon={agent.epsilon:.3f}"
        )

        if episode % SAVE_EVERY == 0:
            checkpoint = CHECKPOINT_DIR / f"mario_dqn_ep{episode:04d}.weights.h5"
            agent.save(checkpoint)

    final_checkpoint = CHECKPOINT_DIR / "mario_dqn_final.weights.h5"
    agent.save(final_checkpoint)
    env.close()


def play_agent(checkpoint_path: str, episodes: int, max_steps: int) -> None:
    env = make_env(render_mode="human")
    action_size = env.action_space.n
    agent = MarioDQNAgent(STATE_SHAPE, action_size)
    agent.load(Path(checkpoint_path))
    agent.epsilon = 0.0  # disable exploration for evaluation

    frame_stacker = FrameStacker(STACK_SIZE)

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        state = frame_stacker.reset(preprocess_frame(obs))
        total_reward = 0.0

        for step in range(1, max_steps + 1):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = frame_stacker.append(preprocess_frame(next_obs))
            total_reward += reward

            if done:
                break

        print(f"Evaluation episode {episode:02d} reward={total_reward:7.2f}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN agent for Super Mario Bros")
    parser.add_argument(
        "--mode",
        choices=["train", "play"],
        default="train",
        help="Select whether to train a new agent or play using a checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAINING_EPISODES,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_PER_EPISODE,
        help="Maximum env steps per episode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to weights to resume training or to play from.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train_agent(
            episodes=args.episodes,
            checkpoint_path=args.checkpoint,
            render=args.render,
            max_steps=args.max_steps,
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when mode='play'")
        play_agent(
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()