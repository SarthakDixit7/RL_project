"""Utility: make_videos.py

Generates MP4 videos from saved Actor models (trained with this repo).

Usage examples:
  python make_videos.py --model-path trainedModels/ALE/Boxing-v5/90/actor_model.keras --episodes 3
  python make_videos.py --models-dir trainedModels --game ALE/Boxing-v5 --every 5 --episodes 2

Notes:
- This script does not modify training code or hyperparameters. It only loads saved actor networks and plays the environment to record frames.
- Requires ffmpeg available on PATH for matplotlib's FFMpegWriter to save MP4 files.
"""

import argparse
import glob
import os
import sys
import time
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model.cnn import ReducedGlorot

# Lazy import for writer support (matplotlib uses ffmpeg under the hood)
from matplotlib.animation import FFMpegWriter


def find_actor_models(models_dir: str, game: str) -> List[str]:
    # game may contain a slash like 'ALE/Boxing-v5'
    game_parts = game.split('/')
    pattern = os.path.join(models_dir, *game_parts, '**', 'actor_model.keras')
    matches = glob.glob(pattern, recursive=True)
    matches = sorted(matches)
    return matches


def pick_models_by_interval(all_models: List[str], every: int = 1, filter_names: List[str] | None = None) -> List[str]:
    if filter_names:
        filtered = [m for m in all_models if any(f in m for f in filter_names)]
        return filtered
    if every <= 1:
        return all_models
    # pick every Nth model
    return [m for i, m in enumerate(all_models) if i % every == 0]


def make_video_for_model(actor_path: str, game: str, output_dir: str, episodes: int, fps: int, seed_start: int = None) -> None:
    print(f"Processing model: {actor_path}")
    # Instantiate an action-only env (ram obs) and a rendering env (rgb)
    action_env = gym.make(game, obs_type='ram')
    render_env = gym.make(game, obs_type='rgb_array', render_mode='rgb_array')

    # Determine actor observation dimensions from ram env
    obs, _ = action_env.reset()
    dimensions = obs.shape
    total_moves = action_env.action_space.n

    # load actor model into a keras model object
    try:
        custom_objects = {"ReducedGlorot": ReducedGlorot}
        loaded = tf.keras.models.load_model(actor_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load model {actor_path}: {e}")
        action_env.close()
        render_env.close()
        return

    # We'll use the loaded keras model directly to produce action probabilities
    def select_action(obs_array: np.ndarray) -> int:
        # obs_array expected shape (1, *dimensions)
        obs_tensor = tf.convert_to_tensor(obs_array, dtype=tf.float32)
        probs = loaded(obs_tensor).numpy()
        probs = probs.reshape(-1)
        action = int(np.random.choice(len(probs), p=probs))
        return action

    model_name = os.path.splitext(os.path.basename(actor_path))[0]
    model_dirname = os.path.basename(os.path.dirname(actor_path))
    # attempt to create an output subdirectory
    safe_name = f"{model_dirname}_{model_name}"
    out_subdir = os.path.join(output_dir, safe_name)
    os.makedirs(out_subdir, exist_ok=True)

    rng = np.random.default_rng(seed_start if seed_start is not None else int(time.time()))

    for ep in range(episodes):
        seed = int(rng.integers(0, 2 ** 31 - 1))

        a_obs, _ = action_env.reset(seed=seed)
        r_obs, _ = render_env.reset(seed=seed)

        # normalize as training does
        a_obs = a_obs / 255.0

        frames = []

        done = False
        truncated = False

        step = 0
        while not (done or truncated):
            state = np.expand_dims(a_obs, axis=0).astype(np.float32)
            action = select_action(state)

            a_obs, reward, done, truncated, _ = action_env.step(action)
            r_obs, reward2, done2, truncated2, _ = render_env.step(action)

            # record current rendered frame
            try:
                frame = render_env.render()
            except Exception:
                # fallback to using returned observation if any
                frame = r_obs

            frames.append(frame)

            # keep norm for policy
            a_obs = a_obs / 255.0
            step += 1
            # safety
            if step > 20000:
                print("Aborting episode due to excessive length")
                break

        # save frames as mp4 using matplotlib FFMpegWriter (requires ffmpeg in PATH)
        out_path = os.path.join(out_subdir, f'episode_{ep+1}.mp4')
        if len(frames) == 0:
            print(f"No frames recorded for {actor_path}, skipping.")
            continue

        fig = plt.figure(figsize=(frames[0].shape[1] / 100.0, frames[0].shape[0] / 100.0), dpi=100)
        plt.axis('off')

        writer = FFMpegWriter(fps=fps)
        try:
            with writer.saving(fig, out_path, dpi=100):
                for frame in frames:
                    plt.imshow(frame.astype('uint8'))
                    plt.axis('off')
                    writer.grab_frame()
            print(f"Saved video -> {out_path}")
        except Exception as e:
            print(f"Failed to save {out_path} (ffmpeg required): {e}")
        finally:
            plt.close(fig)

    action_env.close()
    render_env.close()


def main():
    parser = argparse.ArgumentParser(description='Generate MP4 videos from saved actor models')
    parser.add_argument('--models-dir', default='trainedModels', help='Base models directory to search')
    parser.add_argument('--model-path', default=None, help='Path to specific actor .keras file')
    parser.add_argument('--game', default='ALE/Boxing-v5', help='Gym game id (same as training)')
    parser.add_argument('--output-dir', default='videos', help='Where to write mp4 files')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per model to record')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output MP4')
    parser.add_argument('--every', type=int, default=1, help='If scanning models dir, pick every Nth model')
    parser.add_argument('--filter', type=str, default=None, help='Comma separated names to filter model paths (e.g. 90,100)')
    parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed for deterministic recordings')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_path:
        models = [args.model_path]
    else:
        models = find_actor_models(args.models_dir, args.game)
        if not models:
            print(f"No actor models found under {args.models_dir} for game {args.game}")
            sys.exit(1)

    filter_names = args.filter.split(',') if args.filter else None
    models = pick_models_by_interval(models, every=args.every, filter_names=filter_names)

    if not models:
        print("No models selected after applying filters/intervals.")
        sys.exit(1)

    for model in models:
        make_video_for_model(model, args.game, args.output_dir, episodes=args.episodes, fps=args.fps, seed_start=args.seed)


if __name__ == '__main__':
    main()
