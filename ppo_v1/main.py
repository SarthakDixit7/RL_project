import os
import pickle
import gymnasium as gym
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import argparse
from concurrent.futures import ThreadPoolExecutor
import ale_py
import numpy as np

# from mods
from model.agent import AgentPPO
from model.actor import Actor
from model.critic import Critic

##
## Main running constants
##
GAME = "ALE/Boxing-v5"
EPOCHSPERCYCLE = 3 # 3
# i.e how many sets of trajectories we sample under one 
CYCLES = 2000
EPISODESPERCYCLE = 5
SOLUTIONTHRESHOLD = 90 
COLLECTIONSIZE = 512

# plotting stuff
PLOT = True
FIGURENAME = "Boxing_Not_cos"
LATEX = True
DIAGRAMWIDTH = 397.48499
BORDERTHICKNESS = 0.5
LINETHICKNESS = 0.6
STYLE = "latex_style.mplstyle"

# save stuff
SAVE = True
CHECKPOINTS = False
CHECKPOINTFREQ = 53400
actorPath = f"trainedModels/{GAME}{'/x/check/' if CHECKPOINTS else '/x/'}actor_model"
criticPath = f"trainedModels/{GAME}{'/x/check/' if CHECKPOINTS else '/x/'}critic_model"

# test parameters
TESTFREQ = 2560 * 40
RUNSPERTEST = 20

# video parameters (do not touch training hyperparameters)
# Default set to produce 20 periodic snapshots during a 2000-cycle run (2000 / 100 = 20)
VIDEO_FREQ = 100          # record a video every N cycles (cycles counted from 1)
VIDEO_EPISODES = 1        # episodes per recorded video
VIDEO_FPS = 30            # frames per second for output mp4
VIDEO_DIR = "videos/training"
VIDEO_DETERMINISTIC = True  # use argmax (deterministic) actions for videos

# load model
# replace x with the mean score to load different models
loadModel = False
loadPathActor = f'trainedModels/{GAME}/-177/actor_model'
loadPathCritic = f'trainedModels/{GAME}/-177/critic_model'

##
## PPO hyperparameters
# - Threshold to clip gradients by global norm
# - GAE threashold - has some bias variance tradeoff implications
# - NOT EPS GREEDY - this EPSCLIP the clipping threshold 
# - Return discount duh
TDLAMBDA = 0.90
DISCOUNT = 0.99
EPSCLIP = 0.2
GRADNORM = 0.5
ENTROPY = 0.005

# toggles
USEGAE = True
USEADV = False
USEENTROPY = True

# CNN hyperparameters
CONVOLUTIONS = False
ACTORCONVFILTERS = 32
ACTORDENSEUNITS = 256
CRITICCONVFILTERS = 32
CRITICDENSEUNITS = 512

# DONT use more than 100 for any of the attari games itll go OOM (probably)
BATCHSIZE = 64 # 64


def _ensure_optimizer_built(optimizer, variables):
    if hasattr(optimizer, "built") and not optimizer.built:
        optimizer.build(variables)


def save_training_state(base_actor_path, agent, actor_opt, critic_opt, game_seeds):
    _ensure_optimizer_built(actor_opt, agent.actor.cnn.trainable_variables)
    _ensure_optimizer_built(critic_opt, agent.critic.cnn.trainable_variables)

    training_data = {
        "rolling_mean_history": agent.rolling_mean_history,
        "rolling_mean_store": agent.rolling_mean_history,
        "reward_history": agent.reward_history,
        "step_history": agent.step_history,
        "total_steps": agent.total_steps,
        "total_updates": agent.total_updates,
        "agent_reward_history": agent.reward_history,
        "game_seeds": game_seeds,
        "actor_optimizer_weights": actor_opt.get_weights(),
        "critic_optimizer_weights": critic_opt.get_weights(),
        "actor_optimizer_iterations": int(actor_opt.iterations.numpy()),
        "critic_optimizer_iterations": int(critic_opt.iterations.numpy()),
    }

    file_path = base_actor_path.replace('actor_model', 'training_data.pkl')
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)


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


if __name__ == "__main__":

    # start single env to get dimensions just to make like easier for changing games
    env = gym.make(GAME, obs_type ='ram')
    observation, info = env.reset()
    dimensions = observation.shape
    move_total = env.action_space.n   # type: ignore
    env.close()

    print(f"dims: {dimensions}, action_space: {move_total} ")

    # CLI flags to control video behavior without modifying source 
    parser = argparse.ArgumentParser(description="Train PPO and optionally record periodic videos without changing training params")
    parser.add_argument('--no-video', action='store_true', help='Do not record training videos')
    parser.add_argument('--video-freq', type=int, default=VIDEO_FREQ, help='Record every N cycles (overrides VIDEO_FREQ)')
    parser.add_argument('--video-episodes', type=int, default=VIDEO_EPISODES, help='Episodes per recorded video (overrides VIDEO_EPISODES)')
    parser.add_argument('--video-fps', type=int, default=VIDEO_FPS, help='Frames per second for output mp4')
    parser.add_argument('--video-dir', type=str, default=VIDEO_DIR, help='Output directory for videos')
    vg = parser.add_mutually_exclusive_group()
    vg.add_argument('--video-deterministic', dest='video_deterministic', action='store_true', help='Use deterministic argmax actions for videos')
    vg.add_argument('--video-stochastic', dest='video_deterministic', action='store_false', help='Use stochastic sampling for videos')
    parser.set_defaults(video_deterministic=VIDEO_DETERMINISTIC)
    parser.add_argument('--async-video', action='store_true', help='Run video recording in background threads (periodic snapshots will not block training)')
    parser.add_argument('--no-initial-video', dest='initial_video', action='store_false', help='Skip the initial pre-training video (default: include initial video)')
    parser.set_defaults(initial_video=True)

    args = parser.parse_args()
    video_enabled = not args.no_video
    video_freq = args.video_freq
    video_episodes = args.video_episodes
    video_fps = args.video_fps
    video_dir = args.video_dir
    video_deterministic = args.video_deterministic
    async_video = args.async_video
    initial_video = args.initial_video

    executor = ThreadPoolExecutor(max_workers=2) if async_video and video_enabled else None
    futures = []

    if async_video and video_enabled:
        print(f"Async video recording enabled (saving to {video_dir}, every {video_freq} cycles)")

    # Show expected number of videos for this run (initial + periodic + final)
    if video_enabled:
        expected_periodic = CYCLES // video_freq if video_freq > 0 else 0
        expected_initial = 1 if initial_video else 0
        expected_total = expected_periodic + expected_initial + 1  # include final
        print(f"Video plan: {expected_initial} initial + {expected_periodic} periodic + 1 final = {expected_total} videos (videos saved to {video_dir})")
    else:
        print("Video recording disabled for this run.")

    actor = Actor(
        dimensions=dimensions, 
        regression=False,
        total_moves=move_total,
        conv = CONVOLUTIONS,
        conv_filters=ACTORCONVFILTERS,
        dense_units=ACTORDENSEUNITS,
        eps_clip=EPSCLIP,
        gradnorm=GRADNORM,
        entropy= ENTROPY
    )

    # Optional initial pre-training video (cycle 0)
    if video_enabled and initial_video:
        os.makedirs(video_dir, exist_ok=True)
        init_out = os.path.join(video_dir, 'train_cycle_0.mp4')
        try:
            if executor:
                fut_init = executor.submit(record_video_from_actor, actor, GAME, init_out, video_episodes, video_fps, video_deterministic, seed_start=agent.total_steps if 'agent' in locals() else None)
                futures.append(fut_init)
                def _init_done_callback(fut, path=init_out):
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"Background initial video task failed: {exc}")
                        import traceback
                        traceback.print_exc()
                fut_init.add_done_callback(_init_done_callback)
                print(f"Submitted async initial video task -> {init_out}")
            else:
                print(f"Recording synchronous initial video -> {init_out}")
                record_video_from_actor(actor, GAME, init_out, episodes=video_episodes, fps=video_fps, deterministic=video_deterministic, seed=None)
        except Exception as e:
            print(f"Failed to record initial video: {e}")

    critic = Critic(
        dimensions=dimensions, 
        regression=True,
        total_moves=move_total,
        conv = CONVOLUTIONS,
        conv_filters=CRITICCONVFILTERS,
        dense_units=CRITICDENSEUNITS,
        gradnorm=GRADNORM
    ) 

    # agent constructor
    agent = AgentPPO(
        d_size = EPISODESPERCYCLE,
        collection_size=COLLECTIONSIZE,
        actor= actor,
        critic= critic,
        discount = DISCOUNT, 
        epsilon = EPSCLIP,
        td_lambda = TDLAMBDA
    )
    
    # setup envs to be parallel 
    envs = gym.make_vec(GAME, num_envs=EPISODESPERCYCLE, vectorization_mode="async",obs_type = "ram")


    # lr = 0.00025
    # these settings get ~72 (initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.0025, warmup_steps=1000 , warmup_target=0.00025)
    lr = optimizers.schedules.CosineDecay( initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.05, warmup_steps=1000 , warmup_target=0.00025 )

    act_opt = optimizers.AdamW(learning_rate = lr) # type: ignore
    critic_opt = optimizers.AdamW(learning_rate = lr) # type: ignore
    
    rng = np.random.default_rng()
    # seeds stored in matrix, where [game generation seed , the rest -> epoch shuffle seeds]
    game_seeds = rng.integers(low=0,high=4000000000, size=(CYCLES,EPOCHSPERCYCLE + 1), dtype=np.uint32)

    # try loading models
    if loadModel:
        agent.loadModels(actor_path=loadPathActor, critic_path=loadPathCritic)
        
        filePath = loadPathActor.replace('actor_model', 'training_data.pkl')
        
        try:
            with open(filePath, "rb") as f:
                data = pickle.load(f)

            rolling_hist = data.get("rolling_mean_history") or data.get("rolling_mean_store")
            if rolling_hist is not None:
                agent.rolling_mean_history = rolling_hist

            agent.reward_history = data.get("reward_history", agent.reward_history)
            agent.step_history = data.get("step_history", agent.step_history)
            agent.total_updates = data.get("total_updates", agent.total_updates)
            agent.total_steps = data.get(
                "total_steps",
                agent.step_history[-1] if agent.step_history else agent.total_steps
            )

            loaded_game_seeds = data.get("game_seeds")
            if loaded_game_seeds is not None and len(loaded_game_seeds) > 0:
                game_seeds = loaded_game_seeds

            actor_opt_weights = data.get("actor_optimizer_weights")
            if actor_opt_weights:
                _ensure_optimizer_built(act_opt, actor.cnn.trainable_variables)
                act_opt.set_weights(actor_opt_weights)

            actor_opt_iterations = data.get("actor_optimizer_iterations")
            if actor_opt_iterations is not None:
                act_opt.iterations.assign(actor_opt_iterations)

            critic_opt_weights = data.get("critic_optimizer_weights")
            if critic_opt_weights:
                _ensure_optimizer_built(critic_opt, critic.cnn.trainable_variables)
                critic_opt.set_weights(critic_opt_weights)

            critic_opt_iterations = data.get("critic_optimizer_iterations")
            if critic_opt_iterations is not None:
                critic_opt.iterations.assign(critic_opt_iterations)

            if not agent.reward_history:
                agent.reward_history = [0]
            if not agent.step_history:
                agent.step_history = [agent.total_steps]
            if not agent.rolling_mean_history:
                agent.rolling_mean_history = [0]

            print(f"Loaded training state from {filePath}")
        except FileNotFoundError:
            print("No stored data found")

    ##
    ## Main Training loop
    ##  
    for cycle in range(CYCLES):
        # print(f"\n====================== Cycle {cycle} =========================\n")

        sample_mean, steps = agent.train_cycle(
            envs,
            game_seeds[cycle], 
            actor_opt=act_opt,
            critic_opt=critic_opt,
            epoch_num=EPOCHSPERCYCLE,
            batch_size=BATCHSIZE,
            use_gae = USEGAE,
            use_adv= USEADV,
            use_entropy=USEENTROPY,
        )

        total_updates = ((agent.total_steps/BATCHSIZE) * EPOCHSPERCYCLE)
        # print(f"\n ===>  , total steps this cycle: {steps} ")
        print(f" ===> Steps: {agent.step_history[-1]} | Sample: {cycle} | GradUpdates: {total_updates:.0f} | lr = {critic_opt.learning_rate} | Sample Mean {sample_mean} | Test Rolling Mean: {agent.rolling_mean_history[-1]:.2f}")


        if agent.rolling_mean_history[-1] > SOLUTIONTHRESHOLD:
            print(f"Solution Reached (Mean [-50:] = {agent.rolling_mean_history[-1]:.2f})")
            played_cycles = cycle
            break
            
        if agent.total_steps % TESTFREQ == 0 and agent.total_steps > 0:
            # create a vectorized test env (AgentPPO.__collect_data expects a gym.Env with reset())
            test_env = gym.make_vec(GAME, num_envs=RUNSPERTEST, vectorization_mode="async", obs_type='ram')
            agent.test(test_env, RUNSPERTEST)
            test_env.close()
        
        if agent.total_steps % CHECKPOINTFREQ == 0 and agent.total_steps > 0 and CHECKPOINTS:
            checkpoint_actorPath = actorPath.replace('/check/', f'/{cycle}/')
            checkpoint_criticPath = criticPath.replace('/check/', f'/{cycle}/')
            agent.saveModels(actor_path=checkpoint_actorPath, critic_path=checkpoint_criticPath, checkpoint=True)
            save_training_state(checkpoint_actorPath, agent, act_opt, critic_opt, game_seeds)
            print(f"Checkpoint Models saved to {checkpoint_actorPath} and {checkpoint_criticPath} ")

        agent.clear_data_store()

        # Periodic training snapshot video (configurable / async)
        try:
            if video_enabled and (cycle + 1) % video_freq == 0:
                os.makedirs(video_dir, exist_ok=True)
                out_path = os.path.join(video_dir, f'train_cycle_{cycle+1}.mp4')
                if executor:
                    fut = executor.submit(record_video_from_actor, agent.actor, GAME, out_path, video_episodes, video_fps, video_deterministic, agent.total_steps)
                    futures.append(fut)
                    # Attach a callback to report any exceptions from the background task immediately
                    def _video_done_callback(fut, cyc=cycle+1, path=out_path):
                        try:
                            fut.result()
                        except Exception as exc:
                            print(f"Background video task failed for cycle {cyc}: {exc}")
                            import traceback
                            traceback.print_exc()
                    fut.add_done_callback(_video_done_callback)
                    print(f"Submitted async video task for cycle {cycle+1} -> {out_path}")
                else:
                    print(f"Recording synchronous video for cycle {cycle+1} -> {out_path}")
                    record_video_from_actor(agent.actor, GAME, out_path, episodes=video_episodes, fps=video_fps, deterministic=video_deterministic, seed=agent.total_steps)
        except Exception as e:
            print(f"Failed to record training video for cycle {cycle+1}: {e}")



    # Save models + overwrite checkpoints
    if SAVE:
        rolling = agent.rolling_mean_history[-1]
        if CHECKPOINTS:
            replace = '/x/check/'
        else:
            replace = '/x/'
        actorPath = actorPath.replace(replace, f'/{rolling:.0f}/')
        criticPath = criticPath.replace(replace, f'/{rolling:.0f}/')
        
        agent.saveModels(actor_path=actorPath, critic_path=criticPath, temp=f'{agent.rolling_mean_history[-1]:.0f}', checkpoint=False, saveCheckpoints=CHECKPOINTS)
    
    save_training_state(actorPath, agent, act_opt, critic_opt, game_seeds)

    # Final training video for the last checkpoint
    try:
        if video_enabled:
            os.makedirs(video_dir, exist_ok=True)
            final_out = os.path.join(video_dir, 'final_train.mp4')
            if executor:
                print("Recording final_train.mp4 synchronously (waiting for completion)...")
                fut = executor.submit(record_video_from_actor, agent.actor, GAME, final_out, video_episodes, video_fps, video_deterministic, agent.total_steps)
                fut.result()
            else:
                record_video_from_actor(agent.actor, GAME, final_out, episodes=video_episodes, fps=video_fps, deterministic=video_deterministic, seed=agent.total_steps)
    except Exception as e:
        print(f"Failed to record final training video: {e}")
    finally:
        # if we used an executor, wait for any pending snapshot tasks to finish before exiting
        if executor:
            print("Waiting for background video tasks to finish...")
            for f in futures:
                try:
                    f.result()
                except Exception as ex:
                    print(f"Background video task error: {ex}")
            executor.shutdown(wait=True)


    # plot the trajectory undiscounted return
    if PLOT:
        plt.rcParams['grid.linewidth'] = BORDERTHICKNESS
        plt.rcParams['xtick.major.width'] = BORDERTHICKNESS
        plt.rcParams['ytick.major.width'] = BORDERTHICKNESS
        plt.rcParams['axes.linewidth'] = BORDERTHICKNESS
        plt.rcParams['lines.linewidth'] = LINETHICKNESS

        width = 20
        height = 10
        
        if LATEX: # https://duetosymmetry.com/code/latex-mpl-fig-tips/
            plt.rcParams.update({'text.usetex':True})
            plt.style.use(STYLE)
            pt = 1./72.27
            golden = (1 + 5 ** 0.5) / 2
            width = DIAGRAMWIDTH * pt
            height = width/golden

        plt.figure(figsize = (width,height))
        plt.plot(agent.step_history, agent.reward_history, label = r"PPO $\mu_{D_k}$", alpha = 0.9)
        plt.plot(agent.step_history, agent.rolling_mean_history, label = r"$\text{SMA}_{50}$", linewidth=1)
        plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax= agent.total_steps, colors='r', linestyles='--', linewidth=1, alpha= 0.9)
        plt.xlabel(f"Step total")
        plt.ylabel("Episodic Reward")
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.show()
        plt.savefig(FIGURENAME)