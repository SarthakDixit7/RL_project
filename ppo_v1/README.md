## PPO Atari Boxing (ppo_v1)

This folder trains and evaluates a PPO agent on `ALE/Boxing-v5` using RAM observations. The entrypoint is [ppo_v1/main.py](ppo_v1/main.py).

### Prerequisites
- Python 3.10+ recommended.
- Install system FFmpeg (required for mp4 video writing via `matplotlib` / `imageio-ffmpeg`). On Windows you can install from https://ffmpeg.org and ensure `ffmpeg.exe` is on `PATH`.

### Install Python dependencies
From the repo root (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
```

Key packages: `tensorflow==2.16.1`, `keras==3.4.1`, `gymnasium[atari]==0.29.1`, `ale-py==0.8.1`, `matplotlib`, `numpy`, `imageio`, `imageio-ffmpeg`, `pillow`.

### Files and outputs
- Models, checkpoints, and training metadata save under `ppo_v1/trainedModels/ALE/Boxing-v5/...`.
- Training videos save to `ppo_v1/videos/training` (configurable via CLI).

### Train a **new** model (fresh run)
1. Ensure `loadModel` is `False` near the top of [ppo_v1/main.py](ppo_v1/main.py) (default).
2. Run from this folder:
	```bash
	python main.py
	```
3. Optional video flags (can combine as needed):
	- `--no-video` to disable recording.
	- `--video-freq N` to record every N cycles (default 100).
	- `--video-episodes K` episodes per recorded video (default 1).
	- `--video-fps FPS` output fps (default 30).
	- `--video-dir PATH` change save location.
	- `--video-deterministic` / `--video-stochastic` choose action selection for videos.
	- `--async-video` offload video recording to background threads.
	- `--no-initial-video` skip the pre-training video.

Example (training without videos):
```bash
python main.py --no-video
```

### Resume / continue from an **existing** model
1. Set `loadModel = True` near the top of [ppo_v1/main.py](ppo_v1/main.py).
2. Point `loadPathActor` and `loadPathCritic` to the saved weights you want to resume (e.g. a folder under `ppo_v1/trainedModels/ALE/Boxing-v5/...`). Example:
	```python
	loadPathActor = "trainedModels/ALE/Boxing-v5/-184/actor_model"
	loadPathCritic = "trainedModels/ALE/Boxing-v5/-184/critic_model"
	```
3. Run training as usual:
	```bash
	python main.py
	```
The script restores model weights, optimizer state, and stored training curves if the companion `training_data.pkl` is present. Outputs will continue saving into the configured `actorPath` / `criticPath` templates defined near the top of `main.py`.

### Where to change hyperparameters
Training constants (game, cycles, batch size, network sizes, etc.) are defined at the top of [ppo_v1/main.py](ppo_v1/main.py). Update them before launching a run. Video and saving paths are controlled by the same section.

### Tips
- Running on CPU will be slow; prefer a GPU-enabled TensorFlow install.
- Check disk space in `ppo_v1/trainedModels` and `ppo_v1/videos/training` if running many long jobs.
- If LaTeX plotting fails, set `LATEX = False` in `main.py` or remove the style file reference.
