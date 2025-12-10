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
GAME = "LunarLander-v3" # "ALE/Boxing-v5"
EPOCHSPERCYCLE = 3 # 3
# i.e how many sets of trajectories we sample under one 
CYCLES = 1000
EPISODESPERCYCLE = 10
SOLUTIONTHRESHOLD = 200 # 99

# plotting stuff
PLOT = True
FIGURENAME = "LunarV2"

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

    # setup envs to be parallel - cant use the atari version for ram observation
    # just using parallel envs was giving sample issues? model performance seemed to be worse than running single threaded
    # so separately forcing random seeds per episode for each env
    envs = [gym.make(GAME) for _ in range(EPISODESPERCYCLE)]
    seeds = np.random.randint(0,4000000000,(CYCLES,EPISODESPERCYCLE))


    lr = 0.00025
    act_opt = optimizers.AdamW(learning_rate = lr)
    critic_opt = optimizers.AdamW(learning_rate = lr)
    lr_decay = optimizers.schedules.CosineDecay(0.00025, 3000)

    # note steps is for attempting learning rate scheduling
    rolling_mean_store, rolling_mean, decay_start, total_updates, best_sample_mean, played_cycles = [], 0, 0, 0, 0, 0
        
    for cycle in range(CYCLES):
        print(f" Sample Cycle {cycle} | =============================== | Mean [-10:]: {rolling_mean:.2f} | Total Grad Updates {total_updates:.0f}")
        if best_sample_mean > 0.6*SOLUTIONTHRESHOLD:
            decay_start = cycle
            lr = lr_decay(((cycle - decay_start)/BATCHSIZE)*EPOCHSPERCYCLE)
            print(f"lr Decayed -> {lr}")

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
        print(f" ===> Sample Mean {sample_mean} ")

        if sample_mean >= best_sample_mean:
            best_sample_mean = sample_mean

        if rolling_mean > SOLUTIONTHRESHOLD:
            print(f"Solution Reached (Mean [-10:] = {rolling_mean:.2f})")
            played_cycles = cycle
            break
    
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
        plt.savefig(FIGURENAME)
