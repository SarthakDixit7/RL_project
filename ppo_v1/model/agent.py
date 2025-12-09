import numpy as np
from model.actor import Actor
from model.critic import Critic
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym

## 
## Initial PPO implementation 
## Initial hyperparameter suggenstions taken from https://arxiv.org/pdf/2006.05990
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
            epsilon: float,
            td_lambda: float, 
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
        self.td_lambda = td_lambda

        self.reward_history = []

#
# Training "cycle"
# collects how many trajectories are specified, then stores
#
    def train_cycle(
            self,
            envs, 
            actor_opt, 
            critic_opt, 
            epoch_num,
            batch_size
        ):

        # 1. data collection
        with ThreadPoolExecutor(max_workers = self.d_size) as multi:
            episodes = [multi.submit(self.collect_data, env) for env in envs]
            data = [data.result() for data in episodes]

        for episode in data:
            self.stored_traj["observation"].extend(episode["t_observation"])
            self.stored_traj["reward"].extend(episode["t_reward"])
            self.stored_traj["terminated"].extend(episode["t_terminated"])
            self.stored_traj["truncated"].extend(episode["t_truncated"])
            self.stored_traj["info"].extend(episode["t_info"])
            self.stored_traj["rtg"].extend(episode["t_rtg"])
            self.stored_traj["adv"].extend(episode["t_adv"])
            self.stored_traj["action"].extend(episode["t_action"])
            self.stored_traj["action_prob"].extend(episode["t_action_prob"])
            self.stored_traj["critic_val"].extend(episode["t_critic_vals"])
        
        
        # 2. train the agent
        print("=> Training Agent")
        for epoch in range(epoch_num):

            # print(f" epoch: {epoch}")

            for batch in range(0, len(self.stored_traj["adv"]), batch_size):

                # ONLY CONVERT HERE OTHERWISE GPU MEMORY IS COOKED 
                # gradient tape gets too big if not batch allocated to GPU
                # also means both foward pass, backward pass and tape are all on GPU memory so relitively fast
                #
                # if we could get tf.Dataset working, prefetch and autotune would be vey nice 
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

        # mean = 0
        # if len(self.reward_history)>10:
        mean = np.mean(self.reward_history[-50:])
        return mean


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
        ):

        # For plotting average of samples
        rewards = []

        #
        # run our sampling
        #
        # set current lists
        t_observation, t_reward, t_terminated, t_truncated, t_info, t_action, t_action_prob, t_critic_vals = [],[],[],[],[] ,[], [], []

        # reset env, fill in rest with placeholders
        observation, info = env.reset()

        # keep input to functinoal api CNN happy, need (1,x,y,z) and starting without multiple frames
        # observation = observation[np.newaxis,...,np.newaxis] /255.0
        observation = observation[np.newaxis,:] #/255.0
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
            t_critic_vals.append(self.critic.predict(observation).numpy().item())

            # print(f"State: {steps}, reward: {reward} terminal: {terminated}, truncated: {truncated}, action: {action}, probability {action_prob}")

            observation, reward, terminated, truncated, info = env.step(action)
            # observation = observation[np.newaxis,...,np.newaxis]/255.0
            observation = observation[np.newaxis,:] #/255.0

            # append reward after we observed it so SAR stored at the same index
            t_reward.append(reward)

            total_reward += reward
            steps += 1
        
        #push final
        
        # add to cycle rewards
        rewards.append(total_reward)
        # print(f"Step: Terminal , reward: {reward} terminal: {terminated}, truncated: {truncated}, action: {action}, probability {action_prob}")
        
        # perform both rewards to go + advantage as soon as trajectory done 
        # t_rtg, t_adv, t_critic_vals = self.rtg_adv(t_observation, t_reward)
        t_rtg, t_adv, t_critic_vals = self.rtg_advgeneral(t_observation, t_reward,t_critic_vals)
        # extend experience logs

        print(f"Episode Completed: Reward: {total_reward:.2f} | Steps: {len(t_reward)} ")
        
        data = {
            "t_observation": t_observation,
            "t_reward": t_reward,
            "t_terminated": t_terminated,
            "t_truncated": t_truncated,
            "t_info": t_info,
            "t_rtg": t_rtg,
            "t_adv": t_adv,
            "t_action": t_action,
            "t_action_prob": t_action_prob,
            "t_critic_vals": t_critic_vals,
        }
        
        mean =np.mean(rewards)
        self.reward_history.append(mean)
        # print(f"Iteration mean: {mean}")
        return data

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
    
    def rtg_advgeneral(
            self,
            t_obs,
            t_rw,
            critic_vals,
        ) -> tuple:
        cumulative_reward = 0
        steps = len(t_rw)

        # initialise rtg and advantage lists
        rewards_tg = np.zeros(steps,dtype=np.float32)
        td = np.zeros(steps,dtype=np.float32)
        gae = np.zeros(steps,dtype=np.float32)

        # lots of 0 multiplication will happen before working backward
        # numpy faster than using list tho so, block initialising better?
        lambda_gamma = self.discount * self.td_lambda
        exponents = np.arange(steps)
        bases = np.full(steps,lambda_gamma)
        lg_l = np.power(bases,exponents)

        for time in reversed(range(steps)):
            
            # reward to go
            # as we store s,a,r together at the same index
            reward = t_rw[time]

            rtg = reward + (self.discount * cumulative_reward)

            rewards_tg[time] = rtg
            cumulative_reward = rtg

            disc = 0
            # check if not at last recorded step (terminal doesnt have a value)
            if time != (len(t_rw)-1):
                disc = (self.discount * critic_vals[time + 1])
            
            # Since we store the reward with the action its now
            td[time] = reward + disc - critic_vals[time]

            # calculate elementwise (γλ)^l δV_t+l
            t_gae = np.sum(lg_l * td)

            gae[time] = t_gae

        return rewards_tg, td , gae
