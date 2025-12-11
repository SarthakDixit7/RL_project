import numpy as np
from model.actor import Actor
from model.critic import Critic
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
from operator import itemgetter
from keras import optimizers


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
            batch_size: int,
            epoch_num: int,
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
        self.batch_size: int = batch_size
        self.epoch_num: int = epoch_num

        # for logging progress (if we decide to lol)
        self.best_x: int = 0

        # Models
        self.actor: Actor = actor
        self.critic: Critic = critic

        # parameters
        self.discount: float = discount
        self.epsilon: float = epsilon
        self.td_lambda: float = td_lambda

        # for plotting
        self.reward_history: list = [0]
        self.step_history: list = [0]
        self.total_steps: int = 0

        self.rolling_mean_store: list = [0]
        self.rolling_mean: float = 0
        self.total_updates: int = 0
#
# Training "cycle"
# collects how many trajectories are specified, then stores
#
    def train_cycle(
            self,
            env: gym.Env,
            actor_opt: optimizers.Optimizer, 
            critic_opt: optimizers.Optimizer, 
            use_gae: bool,
            use_adv: bool,
            use_entropy: bool,
        ) -> None:

        self.collect_data(env = env, use_adv=use_adv, use_gae=use_gae)

        samples = len(self.stored_traj["adv"])
        indeces = np.arange(0,samples)

        # 2. train the agent (this was steps 2 and 3 but can do both at the same time)
        for epoch in range(self.epoch_num):

            np.random.shuffle(indeces)

            for batch in range(0, samples, self.batch_size):

                batch_indeces = indeces[batch: (batch+ self.batch_size)].tolist()

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

        rolling_mean = np.mean(self.reward_history[-100:])
        self.rolling_mean = rolling_mean # type: ignore
        self.rolling_mean_store.append(rolling_mean)

        return


#
# 1. Data collection
# Initial training run to fill out D_k trajectories
#

#
# Main collection function
#
    def collect_data(
            self,
            env: gym.Env,
            use_gae: bool,
            use_adv: bool,
        ):

        #
        # run our sampling
        #

        # set current lists
        t_observation, t_reward, t_terminated, t_truncated, t_info, t_action, t_action_prob, t_critic_vals = [],[],[],[],[] ,[], [], []

        # reset env, fill in rest with placeholders
        observation, info = env.reset()

        observation = observation[np.newaxis,:] /255.0
        reward = 0,0
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
            steps += 1

            observation = observation[np.newaxis,:] /255.0

            # append reward after we observed it so S,A,R stored at the same index (makes GAE slightly easier)
            t_reward.append(reward)

            total_reward += reward # type: ignore
            
        # perform both rewards to go + adv as soon as trajectory done
        if use_gae:
            t_rtg, t_adv = self.rtg_advgeneral(t_reward,t_critic_vals)
        elif use_adv:
            t_rtg, t_adv = self.rtg_adv(t_reward,t_critic_vals)

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
        self.stored_traj["critic_val"].extend(t_critic_vals)

        # store total episode reward + steps taken
        self.reward_history.append(total_reward)
        self.total_steps = steps + self.total_steps
        self.step_history.append(self.total_steps)
        self.total_updates = ((self.total_steps/ self.batch_size) * self.epoch_num) # type: ignore

        return steps

#
# GAE calculation
#
    def rtg_advgeneral(
            self,
            t_rw: list,
            critic_vals: list,
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
            t_rw: list, 
            critic_values: list
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
