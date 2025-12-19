import os
import pickle
import gymnasium as gym
from keras import optimizers
import matplotlib.pyplot as plt
import ale_py
import numpy as np
import imageio

# from mods
from ppo_v1.model.agent import AgentPPO
from ppo_v1.model.actor import Actor
from ppo_v1.model.critic import Critic

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

# video recording
RECORD_VIDEO = True
VIDEO_MAX_STEPS = 4000

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
EPSCLIP = 0.15
GRADNORM = 0.5
ENTROPY = 0.01

# toggles
USEGAE = True
USEADV = True
USEENTROPY = True

# CNN hyperparameters
CONVOLUTIONS = False
ACTORCONVFILTERS = 32
ACTORDENSEUNITS = 256
CRITICCONVFILTERS = 32
CRITICDENSEUNITS = 512

# DONT use more than 100 for any of the attari games itll go OOM (probably)
BATCHSIZE = 128


def _ensure_optimizer_built(optimizer, variables):
    if hasattr(optimizer, "built") and not optimizer.built:
        optimizer.build(variables)


def _optimizer_variables(optimizer):
    vars_attr = getattr(optimizer, "variables", None)
    if callable(vars_attr):
        return list(vars_attr())
    if isinstance(vars_attr, (list, tuple)):
        return list(vars_attr)
    return []


def _serialize_optimizer(optimizer):
    # Newer keras optimizers (e.g., AdamW) do not expose get_weights; capture the variable tensors instead.
    return [np.array(v.numpy()) for v in _optimizer_variables(optimizer)]


def _deserialize_optimizer(optimizer, weights):
    opt_vars = _optimizer_variables(optimizer)
    if len(opt_vars) != len(weights):
        print(f"Warning: optimizer variable mismatch (expected {len(opt_vars)}, got {len(weights)})")
        return

    for var, weight in zip(opt_vars, weights):
        var.assign(weight)


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
        "actor_optimizer_weights": _serialize_optimizer(actor_opt),
        "critic_optimizer_weights": _serialize_optimizer(critic_opt),
        "actor_optimizer_iterations": int(actor_opt.iterations.numpy()),
        "critic_optimizer_iterations": int(critic_opt.iterations.numpy()),
    }

    file_path = base_actor_path.replace('actor_model', 'training_data.pkl')
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def record_episode_to_mp4(agent, env_id, output_path, seed=None, max_steps=2000):
    env = gym.make(env_id, obs_type='ram', render_mode='rgb_array')
    observation, info = env.reset(seed=seed)
    frames = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        # normalise like training
        norm_obs = observation / 255.0
        action, _ = agent.actor.choose_action(norm_obs)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    # final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    env.close()

    if frames:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=30)



if __name__ == "__main__":

    # start single env to get dimensions just to make like easier for changing games
    env = gym.make(GAME, obs_type ='ram')
    observation, info = env.reset()
    dimensions = observation.shape
    move_total = env.action_space.n   # type: ignore
    env.close()

    print(f"dims: {dimensions}, action_space: {move_total} ")

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
    
    # setup envs to be parallel - cant use the atari version for ram observation
    # just using parallel envs was giving sample issues? model performance seemed to be worse than running single threaded
    # so separately forcing random seeds per episode for each env, as possibly playing same epsisode seed?
    envs = gym.make_vec(GAME, num_envs=EPISODESPERCYCLE, vectorization_mode="async",obs_type = "ram")


    # lr = 0.00025
    # these settings get ~72 (initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.0025, warmup_steps=1000 , warmup_target=0.00025)
    lr = optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-05,
        decay_steps=200000,
        alpha=0.1,
        warmup_steps=2000,
        warmup_target=3e-04,
    )

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
                _deserialize_optimizer(act_opt, actor_opt_weights)

            actor_opt_iterations = data.get("actor_optimizer_iterations")
            if actor_opt_iterations is not None:
                act_opt.iterations.assign(actor_opt_iterations)

            critic_opt_weights = data.get("critic_optimizer_weights")
            if critic_opt_weights:
                _ensure_optimizer_built(critic_opt, critic.cnn.trainable_variables)
                _deserialize_optimizer(critic_opt, critic_opt_weights)

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
            test_envs = [gym.make(GAME, obs_type ='ram') for _ in range(RUNSPERTEST)]
            agent.test(test_envs)
            for env in test_envs:
                env.close()
        
        if agent.total_steps % CHECKPOINTFREQ == 0 and agent.total_steps > 0 and CHECKPOINTS:
            checkpoint_actorPath = actorPath.replace('/check/', f'/{cycle}/')
            checkpoint_criticPath = criticPath.replace('/check/', f'/{cycle}/')
            agent.saveModels(actor_path=checkpoint_actorPath, critic_path=checkpoint_criticPath, checkpoint=True)
            save_training_state(checkpoint_actorPath, agent, act_opt, critic_opt, game_seeds)
            print(f"Checkpoint Models saved to {checkpoint_actorPath} and {checkpoint_criticPath} ")

        agent.clear_data_store()



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

    if RECORD_VIDEO:
        video_path = actorPath.replace('actor_model', 'training_episode.mp4')
        record_episode_to_mp4(agent, GAME, video_path, seed=int(game_seeds[0][0]), max_steps=VIDEO_MAX_STEPS)


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