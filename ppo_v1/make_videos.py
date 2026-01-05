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
# from matplotlib.animation import FFMpegWriter

# from gymnasium.utils.save_video import save_video


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


def record_video_from_actor(actor, game: str, out_path: str, episodes: int = 1, fps: int = 30, deterministic: bool = True, seed: int | None = None) -> None:
    env = gym.make(game,  obs_type ='ram', render_mode="rgb_array_list")

    observation, info = env.reset(seed=seed)
    print(observation.shape)

    reward, terminated, truncated, info, action, action_prob = 0.0 , False , False , None , 0 , 0

    # run untill the episode ends
    while not terminated and not truncated:
        observation = observation[np.newaxis,:] /255.0

        action, action_prob = actor.choose_action(observation) # type: ignore

        observation, reward, terminated, truncated, info = env.step(int(action))

    save_video(frames= env.render(), video_folder= out_path, fps = env.metadata["render_fps"]) # type: ignore

    env.close()




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
    
def record_video_from_actor(actor, game: str, out_path: str, episodes: int = 1, fps: int = 30, deterministic: bool = True, seed: int | None = None) -> None:
    """Play the environment using the provided Actor instance and save an MP4 to out_path.
    The function uses a RAM env for actor observations (normalized) and an RGB env for rendering frames.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # envs: one for action (ram obs) and one for rendering (rgb)
    action_env = gym.make(game, obs_type='ram')
    # Use obs_type='rgb' for rendering env (Gymnasium expects 'rgb' not 'rgb_array')
    render_env = gym.make(game, obs_type='rgb', render_mode='rgb_array')

    rng = np.random.default_rng(seed if seed is not None else None)

    frames_all = []

    for ep in range(episodes):
        s = int(rng.integers(0, 2 ** 31 - 1)) if seed is None else int(seed + ep)

        a_obs, _ = action_env.reset(seed=s)
        r_obs, _ = render_env.reset(seed=s)

        a_obs = a_obs / 255.0

        frames = []
        done = False
        truncated = False
        step = 0
        while not (done or truncated):
            state = np.expand_dims(a_obs, axis=0).astype(np.float32)

            # get probabilities and choose action (deterministic or sampled)
            try:
                probs = actor.give_action_prob(state).numpy().reshape(-1)
            except Exception:
                probs = actor.cnn(state).numpy().reshape(-1)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))

            a_obs, reward, done, truncated, _ = action_env.step(action)
            r_obs, reward2, done2, truncated2, _ = render_env.step(action)

            try:
                frame = render_env.render()
            except Exception:
                frame = r_obs

            frames.append(frame)

            a_obs = a_obs / 255.0
            step += 1
            if step > 20000:
                print("Aborting episode due to excessive length")
                break

        frames_all.append(frames)

    # Save concatenated frames to out_path using fast imageio-ffmpeg writer when available
    if not frames_all or not any(frames_all):
        print(f"No frames captured for {out_path}")
        action_env.close()
        render_env.close()
        return

    # use first frame to get size
    first_frame = next((f for ep in frames_all for f in ep if f is not None), None)
    if first_frame is None:
        print(f"No valid frames for {out_path}")
        action_env.close()
        render_env.close()
        return

    # Optionally upscale very small frames for better visual quality (keeps aspect ratio)
    def _maybe_upscale(frame: 'np.ndarray', min_width:int=320) -> 'np.ndarray':
        h, w = frame.shape[:2]
        if w >= min_width:
            return frame
        scale = int(min_width / w) + (1 if min_width % w else 0)
        new_w = w * scale
        new_h = h * scale
        try:
            from PIL import Image
            pil = Image.fromarray(frame.astype('uint8'))
            pil = pil.resize((new_w, new_h), Image.BILINEAR)
            return np.asarray(pil)
        except Exception:
            return frame

    # Flatten frames list to per-frame generator
    flat_frames = []
    for ep_frames in frames_all:
        for f in ep_frames:
            if f is None:
                continue
            flat_frames.append(f.astype('uint8'))

    total_frames = len(flat_frames)

    # Try imageio fast path
    try:
        import imageio
        ff_args = ['-preset','ultrafast','-crf','18','-threads','0']
        print(f"Writing {total_frames} frames to {out_path} using imageio-ffmpeg (fast path)")
        writer = imageio.get_writer(out_path, fps=fps, codec='libx264', ffmpeg_params=ff_args)
        for i, frame in enumerate(flat_frames, start=1):
            frame_to_write = _maybe_upscale(frame)
            writer.append_data(frame_to_write)
            if i % 10 == 0 or i == total_frames:
                print(f"  writing frame {i}/{total_frames}")
        writer.close()
        print(f"Saved video -> {out_path} (frames={total_frames})")
    except Exception as e_img:
        print(f"imageio fast path failed ({e_img}), falling back to matplotlib writer")
        import traceback
        traceback.print_exc()
        # Fallback to matplotlib (slower)
        fig = plt.figure(figsize=(first_frame.shape[1] / 100.0, first_frame.shape[0] / 100.0), dpi=100)
        plt.axis('off')
        writer = FFMpegWriter(fps=fps)
        try:
            print(f"Writing {total_frames} frames to {out_path} using matplotlib fallback")
            frame_idx = 0
            with writer.saving(fig, out_path, dpi=100):
                for i, frame in enumerate(flat_frames, start=1):
                    plt.imshow(frame)
                    plt.axis('off')
                    writer.grab_frame()
                    if i % 10 == 0 or i == total_frames:
                        print(f"  writing frame {i}/{total_frames}")
            print(f"Saved video -> {out_path} (frames={total_frames})")
        except Exception as e:
            print(f"Failed to save {out_path} (ffmpeg required?): {e}")
            try:
                # Save a debug frame to help diagnose rendering issues
                debug_frame = first_frame
                if debug_frame is not None:
                    from PIL import Image
                    debug_path = out_path.replace('.mp4', '.debug_frame.png')
                    Image.fromarray(debug_frame.astype('uint8')).save(debug_path)
                    print(f"Saved debug frame -> {debug_path}")
            except Exception as ex_dbg:
                print(f"Failed to save debug frame: {ex_dbg}")
            traceback.print_exc()
        finally:
            plt.close(fig)
    finally:
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
        record_video_from_actor(model, args.game, args.output_dir, episodes=args.episodes, fps=args.fps, seed=args.seed)


if __name__ == '__main__':
    main()
