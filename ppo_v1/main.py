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
EPOCHSPERCYCLE = 10 # 3
# i.e how many sets of trajectories we sample under one 
CYCLES = 2000
EPISODESPERCYCLE = 1 # 1 to keep this simple for plotting
SOLUTIONTHRESHOLD = 90

# plotting stuff
PLOT = True
FIGURENAME = "Boxing_k=1"
LATEX = True
DIAGRAMWIDTH = 397.48499
BORDERTHICKNESS = 0.5
LINETHICKNESS = 0.8
STYLE = "latex_style.mplstyle"

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
        td_lambda = TDLAMBDA,
        batch_size= BATCHSIZE,
        epoch_num= EPOCHSPERCYCLE,
    )

    # lr = 0.00025
    # these settings get ~72 (initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.0025, warmup_steps=1000 , warmup_target=0.00025)
    lr = optimizers.schedules.CosineDecay( initial_learning_rate= 0.0000005 , decay_steps=250000, alpha=0.05, warmup_steps=1000 , warmup_target=0.00025 )

    act_opt = optimizers.AdamW(learning_rate = lr) # type: ignore
    critic_opt = optimizers.AdamW(learning_rate = lr) # type: ignore

    for cycle in range(CYCLES):

        agent.train_cycle(
            env,
            actor_opt=act_opt,
            critic_opt=critic_opt,
            use_gae = USEGAE,
            use_adv= USEADV,
            use_entropy=USEENTROPY,
        )

        agent.clear_data_store()

        if agent.rolling_mean > SOLUTIONTHRESHOLD:
            print(f"Solution Reached (Mean [-50:] = {agent.rolling_mean:.2f}) | Step = {agent.total_steps}")
            break
            
        print(f" Episode: {cycle} |  Step : {agent.step_history[-1]} - 4x: {agent.step_history[-1]*4} | Episode Reward : {agent.reward_history[-1]} GradUpdates: {agent.total_updates:.0f} | lr : {act_opt.learning_rate} | Mean Reward (50 episodes) : {agent.rolling_mean:.2f}")


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
        plt.plot(agent.step_history,agent.reward_history, label = f"Agent Reward")
        plt.plot(agent.step_history,agent.rolling_mean_store, label = r"$\text{SMA}_{50}$", linewidth=1)
        plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax=agent.total_steps, colors='r', linestyles='--', alpha= 0.9)
        plt.xlabel(f"Steps")
        plt.ylabel("Episodic Reward")
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.plot()
        plt.savefig(FIGURENAME)

        env.close()
