import pickle
import gymnasium as gym
from keras import optimizers
import matplotlib.pyplot as plt
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
CYCLES = 2
EPISODESPERCYCLE = 5 
SOLUTIONTHRESHOLD = 80 # 90 -> best target

# plotting stuff
PLOT = True
FIGURENAME = "Boxing_v2"
LATEX = True
DIAGRAMWIDTH = 397.48499
BORDERTHICKNESS = 0.5
LINETHICKNESS = 0.6
STYLE = "latex_style.mplstyle"

# save stuff
SAVE = False
CHECKPOINTS = False
CHECKPOINTFREQ = 250
actorPath = f"trainedModels/{GAME}{'/x/check/' if CHECKPOINTS else '/x/'}actor_model"
criticPath = f"trainedModels/{GAME}{'/x/check/' if CHECKPOINTS else '/x/'}critic_model"

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
        buffer_depth=0, 
        actor= actor,
        critic= critic,
        discount = DISCOUNT, 
        epsilon = EPSCLIP,
        td_lambda = TDLAMBDA
    )
    
    if loadModel:
        agent.loadModels(actor_path=loadPathActor, critic_path=loadPathCritic)
        
        filePath = actorPath.replace('actor_model', 'training_data.pkl')
        
        try:
            with open(filePath, "rb") as f:
                data = pickle.load(f)
                rolling_mean_store = data.get("rolling_mean_store", [])
                mean_store = data.get("mean_store", [])
                step_intervals = data.get("step_intervals", [])
                total_updates = data.get("total_updates", 0)
                best_sample_mean = data.get("best_sample_mean", 0)
        except FileNotFoundError:
            rolling_mean_store, mean_store, step_intervals, rolling_mean, decay_start, total_updates, best_sample_mean, played_cycles = [], [], [], 0, 0, 0, 0, 0

    # setup envs to be parallel - cant use the atari version for ram observation
    # just using parallel envs was giving sample issues? model performance seemed to be worse than running single threaded
    # so separately forcing random seeds per episode for each env, as possibly playing same epsisode seed?
    envs = [gym.make(GAME, obs_type ='ram') for _ in range(EPISODESPERCYCLE)]
    rng = np.random.default_rng()
    seeds = rng.integers(low=0,high=4000000000, size=(CYCLES,EPISODESPERCYCLE), dtype=np.uint32)


    # lr = 0.00025
    # these settings get ~72 (initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.0025, warmup_steps=1000 , warmup_target=0.00025)
    lr = optimizers.schedules.CosineDecay( initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.05, warmup_steps=1000 , warmup_target=0.00025 )

    act_opt = optimizers.AdamW(learning_rate = lr)
    
    critic_opt = optimizers.AdamW(learning_rate = lr)

    # note steps is for attempting learning rate scheduling
    rolling_mean_store, mean_store, step_intervals, rolling_mean, decay_start, total_updates, best_sample_mean, played_cycles = [], [], [], 0, 0, 0, 0, 0
        
    for cycle in range(CYCLES):
        print(f" Sample Cycle {cycle} | =============================== | GradUpdates: {total_updates:.0f} | lr(A,C) = {act_opt.learning_rate} : {critic_opt.learning_rate} | Mean [-50:]: {rolling_mean:.2f}")

        rolling_mean_store.append(rolling_mean)
        agent.clear_data_store()

        rolling_mean, steps, sample_mean = agent.train_cycle(
            envs,
            seeds[cycle], 
            actor_opt=act_opt,
            critic_opt=critic_opt,
            epoch_num=EPOCHSPERCYCLE,
            batch_size=BATCHSIZE,
            use_gae = USEGAE,
            use_adv= USEADV,
            use_entropy=USEENTROPY,
        )
        total_updates += ((steps/BATCHSIZE) * EPOCHSPERCYCLE)
        print(f" ===> Sample Mean {sample_mean} , total steps this cycle: {steps} ")

        if sample_mean >= best_sample_mean:
            best_sample_mean = sample_mean
        
        mean_store.append(sample_mean)
        step_intervals.append(total_updates)

        if rolling_mean > SOLUTIONTHRESHOLD:
            print(f"Solution Reached (Mean [-50:] = {rolling_mean:.2f})")
            played_cycles = cycle
            break
        
        if cycle % CHECKPOINTFREQ == 0 and cycle > 0 and CHECKPOINTS and cycle != CYCLES -1:
            checkpoint_actorPath = actorPath.replace('/check/', f'/{cycle}/')
            checkpoint_criticPath = criticPath.replace('/check/', f'/{cycle}/')
            agent.saveModels(actor_path=checkpoint_actorPath, critic_path=checkpoint_criticPath, checkpoint=True)
            print(f"Checkpoint Models saved to {checkpoint_actorPath} and {checkpoint_criticPath} ")
    
    if SAVE:
        if CHECKPOINTS:
            replace = '/x/check/'
        else:
            replace = '/x/'
        actorPath = actorPath.replace(replace, f'/{rolling_mean:.0f}/')
        criticPath = criticPath.replace(replace, f'/{rolling_mean:.0f}/')
        
        agent.saveModels(actor_path=actorPath, critic_path=criticPath, temp=f'{rolling_mean:.0f}', checkpoint=False, saveCheckpoints=CHECKPOINTS)
        
        # Save training data
        training_data = {
            "rolling_mean_store": rolling_mean_store,
            "mean_store": mean_store,
            "step_intervals": step_intervals,
            "total_updates": total_updates,
            "best_sample_mean": best_sample_mean,
            "agent_reward_history": agent.reward_history,
        }
        
        filePath = actorPath.replace('actor_model', 'training_data.pkl')
        
        with open(filePath, "wb") as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        plt.plot(step_intervals, agent.reward_history, label = r"PPO $\mu_{D_k}$", alpha = 0.9)
        plt.plot(step_intervals, rolling_mean_store, label = r"$\text{SMA}_{50}$", linewidth=1)
        plt.plot(step_intervals, mean_store, label = r"$\mu_{D_k}$", alpha=0.5, linestyle='--')
        plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax=played_cycles, colors='r', linestyles='--', linewidth=1, alpha= 0.9)
        plt.xlabel(f"Collection Iteration " +r"$D_k$, $k$ = "f"{EPISODESPERCYCLE}")
        plt.ylabel("Score")
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.plot()
        plt.savefig(FIGURENAME)