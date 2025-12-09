import gymnasium as gym
from keras import optimizers
import matplotlib.pyplot as plt
import ale_py

# from mods
from model.agent import AgentPPO
from model.actor import Actor
from model.critic import Critic

##
## Main running constants
##
GAME = "LunarLander-v3"  # "ALE/Boxing-v5"
GAMEARGS =  {} #{"obs_type": "ram"}
EPOCHSPERCYCLE = 5 # 3
# i.e how many sets of trajectories we sample under one 
CYCLES = 1000
EPISODESPERCYCLE = 1
SOLUTIONTHRESHOLD = 200 # 99

# plotting stuff
PLOT = True
FIGURENAME = "LunarV1"

##
## PPO hyperparameters
##
# GAE threashold - this has some bias variance tradeoff implications
TDLAMBDA = 0.95
# Return discount
DISCOUNT = 0.99
# THIS IS NOT EPS GREEDY - this is the clipping threshold 
EPSCLIP = 0.2
# Threshold to clip gradients by global norm
GRADNORM = 0.5

# CNN hyperparameters
CONVOLUTIONS = False
ACTORCONVFILTERS = 32
ACTORDENSEUNITS = 128
CRITICCONVFILTERS = 32
CRITICDENSEUNITS = 256

# DONT use more than 100 for any of the attari games itll go OOM (probably)
BATCHSIZE = 64

if __name__ == "__main__":
    # start single env to get dimensions just to make like easier for changing games
    env = gym.make(GAME, kwargs=GAMEARGS)
    observation, info = env.reset(seed=21)
    dimensions = observation.shape
    move_total = env.action_space.n   # type: ignore
    env.close()

    print(f"dims: {dimensions}, action_space: {move_total} ")

    # agent constructor
    agent = AgentPPO(
        d_size = EPISODESPERCYCLE,
        buffer_depth=0, 

        actor=Actor(dimensions=dimensions, 
                    regression=False,
                    total_moves=move_total,
                    conv = CONVOLUTIONS,
                    conv_filters=ACTORCONVFILTERS,
                    dense_units=ACTORDENSEUNITS,
                    ), 

        critic=Critic(dimensions=dimensions, 
                      regression=True,
                      total_moves=move_total,
                      conv = CONVOLUTIONS,
                    conv_filters=CRITICCONVFILTERS,
                    dense_units=CRITICDENSEUNITS
                    ), 

        discount = DISCOUNT, 
        epsilon = EPSCLIP,
        td_lambda = TDLAMBDA
    )

    lr = 0.00025
    # optimizers.schedules.ExponentialDecay(
    #         0.00025,
    #         10000000,
    #         0.95,
    # )
    
    actor_opt = optimizers.Adam(learning_rate = lr, ema_momentum=0.9) # type: ignore
    critic_opt = optimizers.Adam(learning_rate = lr, ema_momentum=0.9) # type: ignore

    # setup envs to be parallel - cant use the atari version for ram observation
    # envs = [gym.make("LunarLander-v3") for _ in range(EPISODESPERCYCLE)]
    envs = [gym.make(GAME, kwargs=GAMEARGS) for _ in range(EPISODESPERCYCLE)]

    rolling_store = []
    # collects for how every many trajectory collection + training sessions
    # cycle seemed like a decent name
    rolling = 0
    for cycle in range(CYCLES):
        print(f"======================================== Cycle {cycle} | Rolling mean: {rolling:.0f} ")
        rolling_store.append(rolling)
        agent.clear_data_store()
        rolling = agent.train_cycle(
            envs, 
            actor_opt, 
            critic_opt , 
            epoch_num=EPOCHSPERCYCLE,
            batch_size=BATCHSIZE
        )

        if rolling > SOLUTIONTHRESHOLD:
            print(f"Solution Reached (Mean [-50:] = {rolling:.2f})")
            break
    
    # plot the trajectory undiscounted return
    if PLOT:
        plt.figure(figsize=(12, 6))
        plt.plot(agent.reward_history, label = "PPO")
        plt.plot(rolling_store, label = "Rolling mean")
        plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax=CYCLES, colors='r', linestyles='-')
        plt.title("Lander Learning Curve")
        plt.xlabel("Learning Cycles")
        plt.ylabel("Return")
        plt.legend()
        plt.grid()
        plt.plot()
        plt.savefig(FIGURENAME)
