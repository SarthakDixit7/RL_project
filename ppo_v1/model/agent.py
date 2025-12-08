import numpy as np
from model.actor import Actor
from model.critic import Critic
import tensorflow as tf

## 
## Initial PPO implementation 
##
## for using keras gradient tape -> https://keras.io/examples/rl/actor_critic_cartpole/
## https://arxiv.org/pdf/2006.05990
##
class AgentPPO:
#
# Constructor
#
    def __init__(
            self ,
            d_size: int ,
            buffer_depth: int ,
            actor: Actor,
            critic: Critic,
            discount: float,
            epsilon: float
        ) -> None:

        # total store of experience - this is done as one big list for each category essentially
        # indecies would be sample number
        self.stored_traj: dict[str, list] = {
            "observation": [] ,
            "reward": [] ,
            "terminated": [] ,
            "truncated": [] ,
            "info": [] ,
            "rtg": [],
            "adv": [],
            "action": [],
            "action_prob": [],
            "critic_val": []
        }

        self.d_size: int  = d_size
        self.buffer_depth: int = buffer_depth

        # for logging progress (if we decide to lol)
        self.best_x: int = 0

        # Models
        self.actor: Actor = actor
        self.critic: Critic = critic

        # parameters (epsilon is for 𝑔(𝜖,𝐴))
        self.discount = discount
        self.epsilon = epsilon

        self.reward_history = []

#
# Training "cycle"
# collects how many trajectories are specified, then stores
#
    def train_cycle(
            self,
            env, 
            actor_opt, 
            critic_opt, 
            epoch_num,
            batch_size
        ) -> None:

        print("=> Collecting Data \n")
        # 1. data collection
        self.collect_data(env)
        
        # 2. train the actor
        print("=> Training Agent")
        for epoch in range(epoch_num):

            # print(f" epoch: {epoch}")

            for batch in range(0, len(self.stored_traj["adv"]), batch_size):

                # ONLY CONVERT HERE OTHERWISE GPU MEMORY IS COOKED - gradient tape gets too big if not batch allocated to GPU
                # again expanding dims to match the concatenated observations, idk if this fixes calc problems but at least its consistent
                # i.e 
                # from 
                # tf.Tensor(x1,x2, ...], shape=(35,), dtype=float32)
                # to
                # tf.Tensor([],[], ...], shape=(35, 1), dtype=float32)
                obs = tf.concat(self.stored_traj["observation"][batch: (batch + batch_size) ], axis=0)
                action_prob_k = tf.expand_dims(tf.convert_to_tensor(self.stored_traj["action_prob"][batch: (batch + batch_size) ]), axis=-1)
                adv_k = tf.expand_dims(tf.convert_to_tensor(self.stored_traj["adv"][batch: (batch + batch_size) ]), axis=-1)
                action_k = tf.expand_dims(tf.convert_to_tensor(self.stored_traj["action"][batch: (batch + batch_size)]), axis=-1)
                rtg = tf.expand_dims(tf.convert_to_tensor(self.stored_traj["rtg"][batch: (batch + batch_size) ]),axis=-1)
                
                self.actor.train(
                    optimiser = actor_opt,
                    obs = obs, 
                    action_prob_k = action_prob_k, 
                    adv_k = adv_k, 
                    action_k = action_k,
                    eps = self.epsilon
                )

                self.critic.train(
                    critic_opt,
                    obs, 
                    rtg, 
                )


#
# 1. Data collection
# Initial training run to fill out D_k trajectories
#

#
# Main collection function
#
    def collect_data(
            self, 
            env
        ) -> None:

        # For plotting average of samples
        rewards = []

        #
        # run our sampling
        #
        for trajectory in range(self.d_size):
            # set current lists
            t_observation, t_reward, t_terminated, t_truncated, t_info, t_action, t_action_prob = [],[],[],[],[] ,[], []

            # reset env, fill in rest with placeholders
            observation, info = env.reset()

            # keep input to functinoal api CNN happy, need (1,x,y,z) and starting without multiple frames
            # observation = observation[np.newaxis,...,np.newaxis] /255.0
            observation = observation[np.newaxis,:]
            reward = 0.0
            terminated = False
            truncated = False
            info = None
            action = 0
            action_prob = 0

            # track total
            total_reward = 0
            steps = 0

            # run untill the episode ends
            while not terminated and not truncated:
                action, action_prob = self.actor.choose_action(observation) # type: ignore

                t_observation.append(observation)
                t_action.append(action)
                t_action_prob.append(action_prob)
                t_terminated.append(terminated)
                t_truncated.append(truncated)
                t_info.append(info)


                # print(f"State: {steps}, reward: {reward} terminal: {terminated}, truncated: {truncated}, action: {action}, probability {action_prob}")

                observation, reward, terminated, truncated, info = env.step(action)
                # observation = observation[np.newaxis,...,np.newaxis]/255.0
                observation = observation[np.newaxis,:]

                # append reward after we observed it so SAR stored at the same index
                t_reward.append(reward)

                total_reward += reward
                steps += 1
            
            #push final
            
            # add to cycle rewards
            rewards.append(total_reward)
            # print(f"Step: Terminal , reward: {reward} terminal: {terminated}, truncated: {truncated}, action: {action}, probability {action_prob}")
            
            # perform both rewards to go + advantage as soon as trajectory done 
            t_rtg, t_adv, t_critic = self.rtg_adv(t_observation, t_reward)
                
            # extend experience logs
            self.stored_traj["observation"].extend(t_observation)
            self.stored_traj["reward"].extend(t_reward)
            self.stored_traj["terminated"].extend(t_terminated)
            self.stored_traj["truncated"].extend(t_truncated)
            self.stored_traj["info"].extend(t_info)
            self.stored_traj["rtg"].extend(t_rtg)
            self.stored_traj["adv"].extend(t_adv)
            self.stored_traj["action"].extend(t_action)
            self.stored_traj["action_prob"].extend(t_action_prob)
            self.stored_traj["critic_val"].extend(t_critic)

            print(f" Iteration: {trajectory} | Reward: {total_reward} | Steps: {len(t_reward)} ")
        
        mean =np.mean(rewards)
        self.reward_history.append(mean)
        print(f"Iteration mean: {mean}")

#
# helpers
#

#
# Just so its not baked into another fn just in case
# dont call this till a training cycle is complete 
#
    def clear_data_store(self) -> None:
        self.stored_traj: dict[str, list] = {
            "observation": [] ,
            "reward": [] ,
            "terminated": [] ,
            "truncated": [] ,
            "info": [] ,
            "rtg": [],
            "adv": [],
            "action": [],
            "action_prob": [],
            "critic_val": []
        }
        
        # print("Store deleted (no getting that back)")

#
# Rewards to go and Advantage in one pass
#

    def rtg_adv(self, t_obs, t_rw):
        cumulative_reward = 0

        # initialise rtg and advantage lists
        rewards_tg = np.zeros(len(t_rw),dtype=np.float32)
        advantage = np.zeros(len(t_obs),dtype=np.float32)

        # get all values from critic 
        critic_values = [self.critic.predict(obs).numpy().item() for obs in t_obs]

        for time in reversed(range(len(t_rw))):
            
            # reward to go
            # as we store s,a,r together at the same index
            reward = t_rw[time]

            rtg = reward + (self.discount * cumulative_reward)

            rewards_tg[time] = rtg
            cumulative_reward = rtg

            disc = 0
            # check if not at last recorded step (terminal doesnt have a value)
            if time != (len(t_rw)-1):
                disc = (self.discount * critic_values[time + 1])
            
            # Since we store the reward with the action its now
            # 𝐴(𝑠𝑡,𝑎𝑡) ≈ 𝑟𝑡+1 + 𝛾𝑉(𝑠𝑡+1) − 𝑉(𝑠𝑡)
            # to
            # 𝐴(𝑠𝑡,𝑎𝑡) ≈ 𝑟𝑡 + 𝛾𝑉(𝑠𝑡+1) − 𝑉(𝑠𝑡)
            advantage[time] = reward + disc - critic_values[time]
        
        # print("\n rtg")
        # print(rewards_tg)
        # print("\n adv")
        # print(advantage)
        # print("\n critic_vals")
        # print(critic_values)

        return rewards_tg, advantage , critic_values
    
    def rtg_advgeneral(self, t_obs, t_rw):
        cumulative_reward = 0

        # initialise rtg and advantage lists
        rewards_tg = np.zeros(len(t_rw),dtype=np.float32)
        advantage = np.zeros(len(t_obs),dtype=np.float32)

        # get all values from critic 
        critic_values = [self.critic.predict(obs).numpy().item() for obs in t_obs]

        for time in reversed(range(len(t_rw))):
            
            # reward to go
            # as we store s,a,r together at the same index
            reward = t_rw[time]

            rtg = reward + (self.discount * cumulative_reward)

            rewards_tg[time] = rtg
            cumulative_reward = rtg

            disc = 0
            # check if not at last recorded step (terminal doesnt have a value)
            if time != (len(t_rw)-1):
                disc = (self.discount * critic_values[time + 1])
            
            # Since we store the reward with the action its now
            # 𝐴(𝑠𝑡,𝑎𝑡) ≈ 𝑟𝑡+1 + 𝛾𝑉(𝑠𝑡+1) − 𝑉(𝑠𝑡)
            # to
            # 𝐴(𝑠𝑡,𝑎𝑡) ≈ 𝑟𝑡 + 𝛾𝑉(𝑠𝑡+1) − 𝑉(𝑠𝑡)
            advantage[time] = reward + disc - critic_values[time]
        
        # print("\n rtg")
        # print(rewards_tg)
        # print("\n adv")
        # print(advantage)
        # print("\n critic_vals")
        # print(critic_values)

        return rewards_tg, advantage , critic_values
