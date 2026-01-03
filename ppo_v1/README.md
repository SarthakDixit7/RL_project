## PPO Atari Boxing

This folder trains and evaluates a PPO agent on `ALE/Boxing-v5` using RAM observations. The entrypoint is [ppo_v1/main.py](ppo_v1/main.py).

### Basic Requirements
- Python 3.10+ recommended
- Install system FFmpeg (required for mp4 video using `matplotlib` / `imageio-ffmpeg`). On Windows you can install from https://ffmpeg.org

### Install Python dependencies
From `requirements.txt` file:

```bash
pip install -r requirements.txt
```
All Libraries used:
```bash
tensorflow==2.16.1
keras==3.4.1
numpy==1.26.4
matplotlib==3.8.2
gymnasium[atari]==0.29.1
ale-py==0.8.1
nes-py==8.2.1
pillow==10.4.0
imageio-ffmpeg==0.4.8
imageio==2.37.2
```

### Files and outputs
- Models, checkpoints, and training metadata save under `ppo_v1/trainedModels/ALE/Boxing-v5/`
- Training videos save to `ppo_v1/videos/training` (configurable via CLI).

### Train a **new** model (fresh run)
1. Ensure `loadModel` is `False` near the top of [ppo_v1/main.py](ppo_v1/main.py) (default).
2. Run this in the terminal from `ppo_v1/` directory:
   
	```bash
	python main.py
	```
3. Video flags:
	- `--no-video` to disable recording.
	- `--video-freq N` to record every N cycles (default 100).
	- `--video-episodes K` episodes per recorded video (default 1).
	- `--video-fps FPS` output fps (default 30).
	- `--video-dir PATH` change save location.
	- `--video-deterministic` / `--video-stochastic` choose action selection for videos.
	- `--async-video` offload video recording to background threads.
	- `--no-initial-video` skip the pre-training video.

### Resume / continue from an **existing** model
1. Set `loadModel = True` near the top of [ppo_v1/main.py](ppo_v1/main.py).
2. Make sure the `loadPathActor` and `loadPathCritic` are linked to a valid directory with the saved weights you want to resume training
   
3. Run training as usual:
	```bash
	python main.py
	```
The programme loads the model weights, optimiser state, and stored training curves if the metadata file `training_data.pkl` is present. Outputs will continue saving into the configured `actorPath` / `criticPath` defined near the top of `main.py`.

### Where to change hyperparameters
Training constants (game, cycles, batch size, network sizes, etc.) are defined at the top of [ppo_v1/main.py](ppo_v1/main.py)

### Changing Environments

To train models in different games/environmets, change the `GAME` variable in [ppo_v1/main.py](ppo_v1/main.py) to whichever environment you want to train for.

### Errors

- If LaTeX is not working for rendering the graphs, set the `LATEX` variable in [ppo_v1/main.py](ppo_v1/main.py) to `False`. This will use the built in `matplotlib` renderer for the graphs instead
