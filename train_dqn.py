import argparse
import logging
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env import make_mario_env


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.obs_shape = observation_shape

    def __len__(self) -> int:
        return len(self.memory)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states, axis=0)).to(device)
        next_states = torch.from_numpy(np.stack(next_states, axis=0)).to(device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones


def epsilon_by_frame(frame_idx: int, eps_start: float, eps_end: float, decay_frames: int) -> float:
    fraction = min(frame_idx / decay_frames, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def save_checkpoint(step: int, q_network: QNetwork, target_network: QNetwork, optimizer: torch.optim.Optimizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": q_network.state_dict(),
            "target_state_dict": target_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / f"checkpoint_{step}.pt",
    )


def train(args: argparse.Namespace) -> dict:
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting DQN training | env=%s | steps=%d | seed=%d | device=%s",
        args.env_id,
        args.total_steps,
        args.seed,
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    env_render_mode = args.render_mode or ("human" if args.render else None)
    env = make_mario_env(
        env_id=args.env_id,
        frame_stack=args.frame_stack,
        reward_clip=args.reward_clip,
        seed=args.seed,
        render_mode=env_render_mode,
    )
    if hasattr(env, "action_space"):
        env.action_space.seed(args.seed)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    q_network = QNetwork(obs_shape[0], action_dim).to(device)
    target_network = QNetwork(obs_shape[0], action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, obs_shape)

    run_name = args.run_name or f"mario_dqn_{args.env_id}_{args.seed}"
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))

    global_step = 0
    best_mean_return = float("-inf")
    episode_counter = 0
    start_time = time.time()
    last_log_time = start_time
    recent_returns: deque[float] = deque(maxlen=args.moving_average_episodes)

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)

    while global_step < args.total_steps:
        epsilon = epsilon_by_frame(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                q_values = q_network(state_t)
                action = int(q_values.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        if args.render:
            env.render()
        done = terminated or truncated
        next_obs = np.asarray(next_obs, dtype=np.float32)

        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        global_step += 1

        if done:
            episode_counter += 1
            moving_mean = math.nan
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                writer.add_scalar("charts/episode_return", ep_return, global_step)
                writer.add_scalar("charts/episode_length", ep_length, global_step)
                recent_returns.append(ep_return)
                if recent_returns:
                    moving_mean = float(np.mean(recent_returns))
                    best_mean_return = max(best_mean_return, moving_mean)
                    writer.add_scalar("charts/best_mean_return", best_mean_return, global_step)
                elapsed = time.time() - start_time
                logger.info(
                    "Episode %d finished | step=%d | return=%.2f | length=%d | epsilon=%.3f | mean_return=%.2f | best_mean=%.2f | elapsed=%.1fs",
                    episode_counter,
                    global_step,
                    ep_return,
                    ep_length,
                    epsilon,
                    moving_mean,
                    best_mean_return if best_mean_return > float("-inf") else float("nan"),
                    elapsed,
                )
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)

        if (
            len(replay_buffer) >= args.learning_starts
            and global_step % args.train_frequency == 0
        ):
            states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size, device)
            with torch.no_grad():
                next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]
                targets = rewards + (1 - dones) * args.gamma * next_q_values

            q_values = q_network(states).gather(1, actions)
            loss = F.smooth_l1_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

            writer.add_scalar("losses/td_loss", loss.item(), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        if global_step % args.target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        if args.checkpoint_dir and global_step % args.checkpoint_interval == 0:
            save_checkpoint(global_step, q_network, target_network, optimizer, Path(args.checkpoint_dir))

        if args.log_interval and (time.time() - last_log_time) >= args.log_interval:
            elapsed = time.time() - start_time
            steps_per_second = global_step / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Progress update | step=%d/%d (%.1f%%) | steps/s=%.1f | epsilon=%.3f | buffer=%d",
                global_step,
                args.total_steps,
                100.0 * global_step / args.total_steps,
                steps_per_second,
                epsilon,
                len(replay_buffer),
            )
            last_log_time = time.time()

    env.close()
    writer.close()

    total_time = time.time() - start_time
    steps_per_second = global_step / total_time if total_time > 0 else 0.0
    logger.info(
        "Training complete | episodes=%d | steps=%d | best_mean_return=%.2f | time=%.1fs | steps/s=%.1f",
        episode_counter,
        global_step,
        best_mean_return if best_mean_return > float("-inf") else float("nan"),
        total_time,
        steps_per_second,
    )

    return {
        "episodes": episode_counter,
        "steps": global_step,
        "best_mean_return": best_mean_return,
        "total_time": total_time,
        "steps_per_second": steps_per_second,
        "tensorboard_run": str(Path(args.log_dir) / run_name),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Super Mario Bros with PyTorch.")
    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v0")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=400_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=8_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=300_000)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-interval", type=float, default=60.0, help="Seconds between progress logs")
    parser.add_argument("--moving-average-episodes", type=int, default=100, help="Episodes for moving average return")
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        default=None,
        help="Render the environment while training",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering during training",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="",
        help="Render mode to use when rendering (e.g. human)",
    )
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
