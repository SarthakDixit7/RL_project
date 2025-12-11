import os
import numpy as np
from model.actor import Actor
from model.critic import Critic
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
from operator import itemgetter
from model.cnn import ReducedGlorot

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

        # parameters
        self.discount = discount
        self.epsilon = epsilon
        self.td_lambda = td_lambda

        # for plotting
        self.reward_history = []
        self.min_rewards = []
        self.max_rewards = []

#
# Training "cycle"
# collects how many trajectories are specified, then stores
#
    def train_cycle(
            self,
            envs, 
            seeds,
            actor_opt, 
            critic_opt, 
            epoch_num,
            batch_size,
            use_gae,
            use_adv,
            use_entropy,
        ):

        # 1. data collection
        with ThreadPoolExecutor(max_workers = self.d_size) as executor:
            # send of envs for data collection + store the futures too
            episodes = [ executor.submit(self.collect_data, env, seeds[i] ,use_gae, use_adv) for i, env in enumerate(envs) ]

            # shove results into list so can sequentially add 
            data = [ env.result() for env in episodes ]
        
        total_steps = 0
        rewards = []

        # add  in single thread to keep indecies matched
        for episode,reward,steps in data:
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
            rewards.append(reward)
            total_steps+= steps        

        mins = np.min(rewards)
        maxs = np.max(rewards)
        self.min_rewards.append(mins)
        self.max_rewards.append(maxs)

        sample_mean = np.mean(rewards)
        self.reward_history.append(sample_mean)


        samples = len(self.stored_traj["adv"])

        indeces = np.arange(0,samples)

        # 2. train the agent (this was steps 2 and 3 but can do both at the same time)
        for epoch in range(epoch_num):

            np.random.shuffle(indeces)

            for batch in range(0, samples, batch_size):

                batch_indeces = indeces[batch: (batch+batch_size)].tolist()

                # https://stackoverflow.com/questions/9106065/python-list-slicing-with-arbitrary-indices
                batch_make = itemgetter(*batch_indeces)

                # ONLY CONVERT HERE OTHERWISE GPU MEMORY IS COOKED (i think)
                # gradient tape gets too big if not batch allocated to GPU
                # also means both foward pass, backward pass and tape are all on GPU memory so relitively fast?
                #
                # if we could get tf.Dataset working, prefetch and autotune would be vey nice 
                obs = tf.concat(batch_make(self.stored_traj["observation"]), axis=0)

                # expanding dims to match the concatenated observations, idk if this fixes calc problems but at least its consistent
                # i.e 
                # from 
                # tf.Tensor(x1,x2, ...], shape=(35,), dtype=float32)
                # to
                # tf.Tensor([],[], ...], shape=(35, 1), dtype=float32)
                action_prob_k = tf.expand_dims(tf.convert_to_tensor(batch_make(self.stored_traj["action_prob"])), axis=-1)
                adv_k = tf.expand_dims(tf.convert_to_tensor(batch_make(self.stored_traj["adv"])), axis=-1)
                action_k = tf.expand_dims(tf.convert_to_tensor(batch_make(self.stored_traj["action"])), axis=-1)
                rtg = tf.expand_dims(tf.convert_to_tensor(batch_make(self.stored_traj["rtg"])),axis=-1)


                self.actor.train(
                    optimiser = actor_opt,
                    obs = obs, 
                    action_prob_k = action_prob_k, 
                    adv_k = adv_k, 
                    action_k = action_k,
                    include_entropy = use_entropy
                )

                self.critic.train(
                    critic_opt,
                    obs, 
                    rtg, 
                )


        rolling_mean = np.mean(self.reward_history[-50:])

        return rolling_mean, total_steps, sample_mean


#
# 1. Data collection
# Initial training run to fill out D_k trajectories
#

#
# Main collection function
#
    def collect_data(
            self,
            env,
            seed,
            use_gae,
            use_adv,
        ):

        #
        # run our sampling
        #

        # set current lists
        t_observation, t_reward, t_terminated, t_truncated, t_info, t_action, t_action_prob, t_critic_vals = [],[],[],[],[] ,[], [], []

        # reset env, fill in rest with placeholders
        observation, info = env.reset(seed= int(seed))

        # keep input to functinoal api CNN happy, need (1,x,y,z) and starting without multiple frames
        # observation = observation[np.newaxis,...,np.newaxis] /255.0
        observation = observation[np.newaxis,:] /255.0
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

            observation, reward, terminated, truncated, info = env.step(action)

            # observation = observation[np.newaxis,...,np.newaxis]/255.0
            observation = observation[np.newaxis,:] /255.0

            # append reward after we observed it so S,A,R stored at the same index (makes GAE slightly easier)
            t_reward.append(reward)

            total_reward += reward
            steps += 1
            
        # perform both rewards to go + adv as soon as trajectory done
        if use_gae:
            t_rtg, t_adv = self.rtg_advgeneral(t_reward,t_critic_vals)
        elif use_adv:
            t_rtg, t_adv = self.rtg_adv(t_reward,t_critic_vals)

        # extend experience logs

        print(f"Episode Completed: Reward: {total_reward:.2f} | Steps: {len(t_reward)} ")
        
        # put in dict to unpack and append outside this fn
        # otherwise cant guarentee order of extend for all lists in parallel
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

        return data, total_reward, steps

#
# GAE calculation
#
    def rtg_advgeneral(
            self,
            t_rw,
            critic_vals,
        ) -> tuple:
        cumulative_reward = 0
        steps = len(t_rw)

        # initialise rtg and advantage lists
        rewards_tg = np.zeros(steps,dtype=np.float32)
        gae = np.zeros(steps,dtype=np.float32)

        # Go back through episode to accumulate reward values
        for time in reversed(range(steps)):
            
            # reward to go
            # as we store s,a,r together at the same index
            reward = t_rw[time]

            rtg = reward + (self.discount * cumulative_reward)

            rewards_tg[time] = rtg
            cumulative_reward = rtg

            delta = 0
            # check if not at last recorded step (terminal doesnt have a value so just skip it )
            if time < steps-1:
                # calculate td value, using same as adv
                # 𝑟𝑡 + 𝛾𝑉(𝑠𝑡+1) − 𝑉(𝑠𝑡)
                delta = reward + (self.discount * critic_vals[time + 1]) - critic_vals[time]
            
            if time< steps -2:
                gae[time] = delta + ((self.discount * self.td_lambda) * gae[time +1])
            else:
                gae[time] = delta

        return rewards_tg, gae

# Works less well than GAE but still here
    def rtg_adv(
            self, 
            t_rw, 
            critic_values
        ) -> tuple :
        cumulative_reward = 0
        steps = len(t_rw)

        # initialise rtg and advantage lists
        rewards_tg = np.zeros(steps,dtype=np.float32)
        advantage = np.zeros(steps,dtype=np.float32)

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

        return rewards_tg, advantage
    
    def saveModels(self, actor_path: str='trainedModels/null/actor_model', critic_path: str='trainedModels/null/critic_model', temp: str='placeholder', checkpoint: bool=False, saveCheckpoints: bool=False) -> None:
        actorFolders = actor_path.rsplit('/')
        if not checkpoint and saveCheckpoints:
            os.rename(f'{actorFolders[0]}/{actorFolders[1]}/x', f'{actorFolders[0]}/{actorFolders[1]}/{temp}')
            
        currnet = ''
        for folder in actorFolders[:-1]:
            currnet += folder
            if not os.path.exists(currnet):
                os.makedirs(currnet)
            currnet += '/'
        
        currnet = ''
        criticFolders = critic_path.rsplit('/')
        for folder in criticFolders[:-1]:
            currnet += folder
            if not os.path.exists(currnet):
                os.makedirs(currnet)
            currnet += '/'
        
        self.actor.cnn.save(f'{actor_path}.keras')
        self.critic.cnn.save(f'{critic_path}.keras')
        print(f" Models saved to {actor_path} and {critic_path} ")
        
    def loadModels(self, actor_path: str='trainedModels/actor_model', critic_path: str='trainedModels/critic_model') -> None:
        custom_objects = {"ReducedGlorot": ReducedGlorot}
        self.actor.cnn = tf.keras.models.load_model(f'{actor_path}.keras', custom_objects=custom_objects)
        self.critic.cnn = tf.keras.models.load_model(f'{critic_path}.keras', custom_objects=custom_objects)
        print(f" Models loaded from {actor_path} and {critic_path} ")
    
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
