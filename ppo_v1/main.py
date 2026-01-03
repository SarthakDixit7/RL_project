import os
import pickle
import gymnasium as gym
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import argparse
import utils as utils
from make_videos import record_video_from_actor
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
COLLECTIONSIZE = 1787

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
CHECKPOINTS = True
CHECKPOINTFREQ = COLLECTIONSIZE * EPISODESPERCYCLE * 10
actorPath = f"trainedModels/{GAME}{'/x/check/' if CHECKPOINTS else '/x/'}actor_model"
criticPath = f"trainedModels/{GAME}{'/x/check/' if CHECKPOINTS else '/x/'}critic_model"

# test parameters
TESTFREQ = COLLECTIONSIZE * EPISODESPERCYCLE * 10
RUNSPERTEST = 10

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

    # Optional initial pre-training video (cycle 0)
    if video_enabled and initial_video:
        os.makedirs(video_dir, exist_ok=True)
        init_out = os.path.join(video_dir, 'train_cycle_0.mp4')
        try:
            if executor:
                fut_init = executor.submit(record_video_from_actor, actor, GAME, init_out, video_episodes, video_fps, video_deterministic, agent.total_steps if 'agent' in locals() else None)
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
    

    # setup envs to be parallel 
    envs = gym.make_vec(GAME, num_envs=EPISODESPERCYCLE, vectorization_mode="async",obs_type = "ram")

    # Initialize bestScore for tracking best sample_mean
    bestScore = float('-inf')


    # lr = 0.00025
    # these settings get ~72 (initial_learning_rate= 0.0000005 , decay_steps=200000, alpha=0.05, warmup_steps=1000 , warmup_target=0.00025)
    lr = optimizers.schedules.CosineDecay( initial_learning_rate= 0.0000005 , decay_steps=200000, alpha=0.05, warmup_steps=1000 , warmup_target=0.00025 )
    act_opt = optimizers.AdamW(learning_rate = lr) # type: ignore
    critic_opt = optimizers.AdamW(learning_rate = lr) # type: ignore
    
    rng = np.random.default_rng()
    # seeds stored in matrix, where [game generation seed , the rest -> epoch shuffle seeds]
    # game_seeds = rng.integers(low=0,high=4000000000, size=(CYCLES,EPOCHSPERCYCLE + 1), dtype=np.uint32)
 

    # try loading models

    if loadModel:
        game_seeds = utils.loadModel(agent, loadPathActor, loadPathCritic, act_opt, critic_opt, actor, critic)

        # Load game_seeds from training_data.pkl if available
        filePath = loadPathActor.replace('actor_model', 'training_data.pkl')
        try:
            with open(filePath, "rb") as f:
                data = pickle.load(f)
            loaded_game_seeds = data.get("game_seeds")
            if loaded_game_seeds is not None and len(loaded_game_seeds) > 0:
                game_seeds = loaded_game_seeds
        except Exception as e:
            print(f"Warning: Could not load game_seeds from {filePath}: {e}")

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
        
        utils.save(
            actorPath, 
            criticPath, 
            CHECKPOINTS, 
            agent, 
            CHECKPOINTFREQ, 
            cycle, 
            agent.rolling_mean_history[-1], 
            act_opt, 
            critic_opt, 
            game_seeds, 
            False, 
            asBest=bestScore<sample_mean
        )
        agent.clear_data_store()

        # Update bestScore if current sample_mean is better
        if sample_mean > bestScore:
            bestScore = sample_mean

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
        utils.save(
            actorPath, 
            criticPath, 
            CHECKPOINTS, 
            agent, 
            CHECKPOINTFREQ, 
            cycle, 
            agent.rolling_mean_history[-1], 
            act_opt, 
            critic_opt, 
            game_seeds,
            True,
            asBest=bestScore<sample_mean
        )

        # Update bestScore for final save
        if sample_mean > bestScore:
            bestScore = sample_mean
    
    utils.save_training_state(actorPath, agent, act_opt, critic_opt, game_seeds)

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
        utils.plotGraph(
            BORDERTHICKNESS, 
            LINETHICKNESS, 
            LATEX, 
            STYLE, 
            DIAGRAMWIDTH, 
            SOLUTIONTHRESHOLD, 
            FIGURENAME, 
            agent
        )