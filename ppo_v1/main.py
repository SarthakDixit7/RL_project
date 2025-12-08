import gymnasium as gym
from keras import optimizers
import matplotlib.pyplot as plt
import ale_py

# from mods
from model.agent import AgentPPO
from model.actor import Actor
from model.critic import Critic

#
# Main running constants
#
EPOCHSPERCYCLE = 5
CYCLES = 1000
PLOT = True
EPISODENUM = 5
SOLUTIONTHRESHOLD = 200

# network hyperparameters
CONVOLUTIONS = False
ACTORCONVFILTERS = 32
ACTORDENSEUNITS = 128
CRITICCONVFILTERS = 32
CRITICDENSEUNITS = 256

# DONT use more than 100 for any of the attari games itll go OOM 
BATCHSIZE = 64

if __name__ == "__main__":
    # start env
    env = gym.make("ALE/Boxing-v5"
                #     ,obs_type="grayscale"
                    ,obs_type="ram"
                #    , render_mode="human"
    )

    observation, info = env.reset(seed=21)

    dimensions = observation.shape
    move_total = env.action_space.n   # type: ignore

    print(f"dims: {dimensions}, action_space: {move_total} ")

    # note eps is clip NOT explore
    agent = AgentPPO(
        d_size=EPISODENUM,
        buffer_depth=25, 
        actor=Actor(dimensions=dimensions, 
                    regression=False,
                    total_moves=move_total,
                    conv = CONVOLUTIONS,
                    conv_filters=ACTORCONVFILTERS,
                    dense_units=ACTORDENSEUNITS
                    ), 
        critic=Critic(dimensions=dimensions, 
                      regression=True,
                      total_moves=move_total,
                      conv = CONVOLUTIONS,
                    conv_filters=CRITICCONVFILTERS,
                    dense_units=CRITICDENSEUNITS
                    ), 
        discount=0.99, 
        epsilon=0.2
    )
    
    actor_opt = optimizers.AdamW(learning_rate = 0.00025)
    critic_opt = optimizers.AdamW(learning_rate = 0.00025)

    # collects for how every many trajectory collection + training sessions
    # cycle seemed like a decent name
    for cycle in range(CYCLES):
        print(f" ------------- Cycle {cycle} ------------- ")
        agent.clear_data_store()

        reward = agent.train_cycle(
            env, 
            actor_opt, 
            critic_opt , 
            epoch_num=EPOCHSPERCYCLE,
            batch_size=BATCHSIZE
        )
    
    # plot the trajectory undiscounted return
    if PLOT:
        plt.figure(figsize=(12, 6))
        plt.plot(agent.reward_history, label = "PPO")
        plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax=CYCLES, colors='r', linestyles='-')
        plt.title("Lander Learning Curve")
        plt.xlabel("Learning Cycles")
        plt.ylabel("Return")
        plt.legend()
        plt.grid()
        plt.show()
