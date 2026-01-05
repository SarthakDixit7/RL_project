# RL_project

## Quick Start — Step by Step (recommended order)

Follow these in order to get training and video recording working reliably.

1) Create & activate a venv

- Windows (PowerShell):
```powershell
python -m venv .venv
& .venv/Scripts/Activate.ps1
```
- macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install Python requirements

```bash
pip install -r requirements.txt
```

3) Install system ffmpeg (required to save MP4s)

- Chocolatey (Windows, admin):
```powershell
choco install ffmpeg -y
refreshenv
ffmpeg -version
```
- winget (Windows):
```powershell
winget install --exact --id Gyan.FFmpeg
ffmpeg -version
```
- Manual (no admin): download a build, unzip to `C:\ffmpeg` and add `C:\ffmpeg\bin` to your User PATH.

Verify ffmpeg:
```bash
ffmpeg -version
```

4) Install Atari ROMs (AutoROM)

```bash
pip install "gymnasium[atari,accept-rom-license]" autorom
# then
autorom --accept-license
# or from venv explicitly
.venv\Scripts\autorom.exe --accept-license
```

5) Verify the Boxing env is available

```bash
python - <<'PY'
import gymnasium as gym
try:
    env = gym.make("ALE/Boxing-v5", obs_type="ram")
    print("OK: Boxing available")
    env.close()
except Exception as e:
    print("ERROR:", e)
PY
```

6) Run training with async video snapshots (defaults: 20 periodic snapshots + 1 initial + 1 final = 22 videos for a 2000-cycle default run — i.e. every 100 cycles, 1 episode per snapshot). Use `--no-initial-video` to skip the initial pre-training snapshot.

```bash
python ppo_v1/main.py --async-video
```

---

## 🎬 Recording Training Videos (Quick Guide)

This project can record MP4 videos of training snapshots and the final checkpoint without changing any training hyperparameters.

---

## 🔧 Setup (venv, deps, ROMs, ffmpeg)

1) Create & activate a virtual environment

- Windows (PowerShell):
```powershell
python -m venv .venv
& .venv/Scripts/Activate.ps1
```

- macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install Python dependencies

```bash
pip install -r requirements.txt
```

Tip: the repo includes `imageio-ffmpeg` in `requirements.txt` but that package does not always expose a globally available `ffmpeg` executable — you still need a system ffmpeg binary on PATH or point matplotlib to the provided binary.

---

## 🕹️ Atari ROMs (required for ALE environments like `ALE/Boxing-v5`)

Gymnasium does not distribute Atari ROMs. Use AutoROM to download and install them in your active venv.

- Install AutoROM and the Gymnasium Atari extra:
```bash
pip install "gymnasium[atari,accept-rom-license]" autorom
```

- Run the AutoROM CLI to download + install (auto-accept license):
```bash
# preferred: run the script installed into the venv
autorom --accept-license
# or run the script directly from the venv (PowerShell)
.venv\Scripts\autorom.exe --accept-license
```

Notes:
- On some systems `python -m autorom` may not work; using the `autorom` console script (shown above) is reliable.
- If you already have ROM files, import them with `ale-import-roms /path/to/roms`.

Verify Boxing is available:
```bash
python - <<'PY'
import gymnasium as gym
try:
    env = gym.make("ALE/Boxing-v5", obs_type="ram")
    print("OK: Boxing available")
    env.close()
except Exception as e:
    print("ERROR:", e)
PY
```

---

## 🎞️ FFmpeg (required to write MP4s)

Matplotlib's `FFMpegWriter` needs an ffmpeg executable. Here are simple install options.

Option A — Chocolatey (recommended on Windows, requires admin):
```powershell
# Run PowerShell as Administrator
choco install ffmpeg -y
refreshenv
ffmpeg -version
```

Option B — winget (if available):
```powershell
winget install --exact --id Gyan.FFmpeg
ffmpeg -version
```


## ▶️ How to run training with video recording

- Default (videos enabled unless you pass `--no-video`):
```bash
python ppo_v1/main.py --async-video
```
This runs training and records periodic snapshots asynchronously (default: every 20 cycles, 1 episode per snapshot).

Useful flags
- `--no-video` — disable recordings
- `--async-video` — run periodic recordings in background threads (non-blocking)
- `--video-freq N` — record every N cycles (default: 20)
- `--video-episodes N` — episodes per recorded video (default: 1)
- `--video-fps N` — frames per second (default: 30)
- `--video-dir DIR` — output directory (default: `videos/training`)
- `--video-deterministic` / `--video-stochastic` — deterministic (argmax) vs sampled actions (default: deterministic)

What to expect
- Periodic snapshots saved as `videos/training/train_cycle_{cycle}.mp4`
- Final checkpoint video saved as `videos/training/final_train.mp4`
- With `--async-video`, snapshot tasks are queued and the script waits for them to finish on exit

---

## 🎥 Generate videos from saved models (offline)

To convert saved `actor_model.keras` files into MP4s without running training:
```bash
python ppo_v1/make_videos.py --model-path trainedModels/ALE/Boxing-v5/90/actor_model.keras --episodes 3
# or scan a directory and choose every Nth model
python ppo_v1/make_videos.py --models-dir trainedModels --game ALE/Boxing-v5 --every 5 --episodes 2
```

---

## ⚠️ Troubleshooting

- "We're Unable to find the game \"Boxing\"": ensure AutoROM has installed ROMs; run `autorom --accept-license` in the same venv used to run training. If autorom isn't on PATH, run `.venv\Scripts\autorom.exe --accept-license`.
- Test that Boxing is available using the Python snippet under the ROMs section.
- "Failed to save ... (ffmpeg required?)": install ffmpeg and verify `ffmpeg -version` works or set `mpl.rcParams['animation.ffmpeg_path']` to a valid binary.
- If background snapshot tasks are submitted but you see no MP4s, run training synchronously (without `--async-video` and with `--video-freq 1`) to see immediate errors.

---

## Notes

- The video helpers and CLI flags are implemented to **not** modify any training hyperparameters (learning rate, epochs, etc.). Videos are produced from the current in-memory `agent.actor` model during training or by loading saved actor models offline.

---

Happy training! 🏆


### 🔧 Setup (venv, requirements, ROMs, ffmpeg)

1. Create & activate a virtual environment

   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     & .venv/Scripts/Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

2. Install Python deps

   ```bash
   pip install -r requirements.txt
   ```

   Note: If you plan to use Atari environments and you haven't accepted the ROM license, install the extra and import ROMs (see next step).

3. Install / import Atari ROMs (required for ALE environments like `ALE/Boxing-v5`)

   - Import ROMs (auto-accept license):
     ```bash
     python -m autorom --accept-license
     ```

4. Ensure ffmpeg binary is installed and on your PATH (required to save MP4s)

   - Windows (Chocolatey):
     ```powershell
     choco install ffmpeg
     ```
   - macOS (Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - Verify:
     ```bash
     ffmpeg -version
     ```

---

## ▶️ How to run training with video recording

- Default (videos enabled by default when you don't pass `--no-video`):
  ```bash
  python ppo_v1/main.py --async-video
  ```
  This runs training and records periodic snapshots asynchronously (default: every 20 cycles, 1 episode per snapshot).

- flags:
  - `--no-video` — disable all recording
  - `--async-video` — record periodic videos in background threads (does not block training)
  - `--video-freq N` — record every N cycles (default: 20)
  - `--video-episodes N` — episodes per recorded video (default: 1)
  - `--video-fps N` — frames per second for MP4 (default: 30)
  - `--video-dir DIR` — output directory (default: `videos/training`)
  - `--video-deterministic` / `--video-stochastic` — deterministic (argmax) or sampled actions in recorded videos (default: deterministic)

- What to expect:
  - Periodic snapshots saved as `videos/training/train_cycle_{cycle}.mp4`
  - Final checkpoint video saved as `videos/training/final_train.mp4`
  - If `--async-video` is used, snapshot tasks are queued in background threads (the script waits for them to finish on exit)

---

## 🎥 Generate videos from saved models (offline)

If you want to convert saved `actor_model.keras` files into MP4s without running training, use the utility:

```bash
python ppo_v1/make_videos.py --model-path trainedModels/ALE/Boxing-v5/90/actor_model.keras --episodes 3
# or scan a directory and choose every Nth model
python ppo_v1/make_videos.py --models-dir trainedModels --game ALE/Boxing-v5 --every 5 --episodes 2
```

The script supports filtering (`--filter 90,100`), fps (`--fps`), and output dir (`--output-dir`).

---

## ⚠️ Troubleshooting

- Error: `We're Unable to find the game "Boxing"` — run `python -m autorom --accept-license` or `ale-import-roms` and ensure ROMs are installed.
- Error saving MP4: Make sure `ffmpeg` is installed and on PATH (see step 4 above).
- If you want to disable video recording entirely for long experiments, use `--no-video`.

---

## Utilised Code
- Correctly using TF gradient tape to update weights
  - Reference on using the function contained in Keras documentation
  - tape.gradient and .apply gradients
  - https://keras.io/examples/rl/actor_critic_cartpole/

- Slicing for arbitrary indeces
  - Again showed how to use itemgetter, used to get our batch indeces
  - https://stackoverflow.com/questions/9106065/python-list-slicing-with-arbitrary-indices

- Custom weights
  - Used as reference on layer weight modification in conjunction with keras docs
  -https://datascience.stackexchange.com/questions/19019/custom-weight-initialization-in-keras

- Video Saving
  - Showed how to use the save_video function using rgb_list
  - https://stackoverflow.com/questions/77042526/how-to-record-and-save-video-of-gym-environment

- Tutorial for Graph Plotting with latex
  - used the ratios provided
  - https://duetosymmetry.com/code/latex-mpl-fig-tips/