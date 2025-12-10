import gymnasium as gym
from keras import optimizers
import matplotlib.pyplot as plt
import ale_py
import numpy as np
import tensorflow as tf

# from mods
from model.agent import AgentPPO
from model.actor import Actor
from model.critic import Critic

##
## Main running constants
##
GAME = "LunarLander-v3" # "ALE/Boxing-v5"
EPOCHSPERCYCLE = 3 # 3
# i.e how many sets of trajectories we sample under one 
CYCLES = 5
EPISODESPERCYCLE = 10
SOLUTIONTHRESHOLD = 200 # 99

# plotting stuff
PLOT = True
FIGURENAME = "LunarV2"

# save stuff
SAVE = True
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
ENTROPY = 0.0005

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
BATCHSIZE = 64

class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initLR, warmupSteps, totalSteps):
        super().__init__()
        self.lr = initLR
        self.initLR = initLR
        self.warmupSteps = warmupSteps
        self.totalSteps = max(1, totalSteps)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        w = tf.cast(self.warmupSteps, tf.float32)
        t = tf.cast(self.totalSteps, tf.float32)

        warmup = tf.minimum(1.0, step / tf.maximum(1.0, w))
        decaySteps = tf.maximum(1.0, t - w)
        decayProgress = tf.clip_by_value((step - w) / decaySteps, 0.0, 1.0)
        decay = 1.0 - decayProgress
        
        self.lr = self.initLR * warmup * decay

        return self.lr

    def __str__(self):
        return f"CustomLRSchedule(initLR={self.initLR}, warmupSteps={self.warmupSteps}, totalSteps={self.totalSteps}), currentLR={self.lr}"

def makeLRScheduler(initLR, warmupSteps, totalUpdates):
    return CustomLRSchedule(initLR, warmupSteps, totalUpdates)

if __name__ == "__main__":
    # start single env to get dimensions just to make like easier for changing games
    env = gym.make(GAME)
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

    # setup envs to be parallel - cant use the atari version for ram observation
    # just using parallel envs was giving sample issues? model performance seemed to be worse than running single threaded
    # so separately forcing random seeds per episode for each env
    envs = [gym.make(GAME) for _ in range(EPISODESPERCYCLE)]
    rng = np.random.default_rng()
    seeds = rng.integers(low=0,high=4000000000, size=(CYCLES,EPISODESPERCYCLE), dtype=np.uint32)
    
    lr = 0.00025
    totalUpdates = CYCLES * EPOCHSPERCYCLE or 1 * EPOCHSPERCYCLE or 1
    warmupSteps = min(1000, totalUpdates // 100)

    totalSteps = max(1, totalUpdates)
    lrSchedule = makeLRScheduler(lr, warmupSteps, totalSteps)
    
    weightDecay = 1e-4
    
    act_opt = optimizers.AdamW(learning_rate = lrSchedule, weight_decay=weightDecay)
    critic_opt = optimizers.AdamW(learning_rate = lrSchedule, weight_decay=weightDecay)

    # note steps is for attempting learning rate scheduling
    rolling_mean_store, rolling_mean, decay_start, total_updates, best_sample_mean, played_cycles = [], 0, 0, 0, 0, 0
    
    doKlEStopping = False
        
    for cycle in range(CYCLES):
        print(f" Sample Cycle {cycle} | =============================== | Mean [-10:]: {rolling_mean:.2f} | Total Grad Updates {total_updates:.0f}")

        rolling_mean_store.append(rolling_mean)
        agent.clear_data_store()
        
        if (cycle / CYCLES) > 0.15 and not doKlEStopping:
            doKlEStopping = True
            print(" KL Early Stopping Enabled ")

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
            use_kl_early_stopping=doKlEStopping,
        )
        total_updates += ((steps/BATCHSIZE) * EPOCHSPERCYCLE)
        print(f" ===> Sample Mean {sample_mean} ")

        if sample_mean >= best_sample_mean:
            best_sample_mean = sample_mean

        if rolling_mean > SOLUTIONTHRESHOLD:
            print(f"Solution Reached (Mean [-10:] = {rolling_mean:.2f})")
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
    
    # plot the trajectory undiscounted return
    if PLOT:
        plt.figure(figsize=(12, 6))
        plt.plot(agent.reward_history, label = "PPO")
        plt.plot(rolling_mean_store, label = "Rolling mean")
        plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax=played_cycles, colors='r', linestyles='-')
        plt.title("Lander Learning Curve")
        plt.xlabel("Learning Cycles")
        plt.ylabel("Return")
        plt.legend()
        plt.grid()
        plt.plot()
        plt.savefig(actorPath.replace('actor_model', FIGURENAME + '_learning_curve.png'))
