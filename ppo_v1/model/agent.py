from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from model.actor import Actor
from model.critic import Critic
import tensorflow as tf
from multiprocessing import Pool, TimeoutError
import gymnasium as gym
from operator import itemgetter
from model.cnn import ReducedGlorot

## 
## Initial PPO implementation 
## Initial hyperparameter suggenstions taken from https://arxiv.org/pdf/2006.05990
##
class AgentPPO:
    """
    Main PPO agent Class with GAE + Entropy

    Attributes
    ----------
    stored_traj: dict[str, list]
        - All relevant stored trajectory information. These are stored as lists and are extended with each episodes complete trajectory
        - Note: correct indecies are currently ensured by extending single threaded after all episodes are complete
        - {
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
    d_size: int
        - Number of trajectories samples during each iteration of k (or in our case cycles)
    collection_size: int
        - Numper of steps played by each env before updating
    actor: Actor
        - Constructed Actor class containing its network and update methods
    critic: Critic
        - Constructed Critic class containing its network and update methods
    discount: float 
        - Designated discount 𝛾 value for calculating dicounted sum of returns
    epsilon: float
        - Clipping threshold designated in PPO clip
    td_lambda: float
        - GAE weighting term
    reward_history: list
        - Total mean episodic reward store, across samples collected each cycle
    step_history: list
        - Cumulative sum of all played steps after each cycle correspoding to the reward mean
    total_steps: int
        - Counter to hold cumulative steps 
    rolling_mean_history: list
        - complete rolling mean history at each step_value
    total_updates: int
        - Total gradient updates observed

    Methods
    -------   
    train_cycle
        - Main training cycle for the model (the loop indexed by k in the psudocode)
        - Available toggles for GAE or TD(1)
        - Experience collection is multithreaded with number of collected trajectories dependant on self.d_size
    save_models
        - Function to store curent weights in .keras file
    load_models
        - Function to set current Actor and Critic to some stored models
        - This is useful if training stagnates and the learning rate needs to be manually modified OR for testing purposes
    clear_data
        - Resets the stored_traj attribute to allow for a new set of training data to be collected
    """

    def __init__(
            self ,
            d_size: int ,
            collection_size: int,
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
        self.collection_size: int = collection_size

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
        self.rolling_mean_history: list = [0]
        self.total_updates: int = 0


##################################################################################################################################################
## Public Methods
##################################################################################################################################################

    def train_cycle(
            self,
            env, 
            seed,
            actor_opt, 
            critic_opt, 
            epoch_num,
            batch_size,
            use_gae,
            use_adv,
            use_entropy,
        ) -> tuple :

        mean_reward, total_steps = self.__collect_data(env=env, seed=seed[0] ,use_gae=use_gae)

        indeces = np.arange(0,self.collection_size)

        # 2. train the agent (this was steps 2 and 3 but can do both at the same time)
        for epoch in range(epoch_num):
            np.random.seed = seed[epoch + 1]
            np.random.shuffle(indeces)

            for batch in range(0, self.collection_size, batch_size):

                batch_indeces = indeces[batch: (batch+batch_size)].tolist()

                # https://stackoverflow.com/questions/9106065/python-list-slicing-with-arbitrary-indices
                batch_make = itemgetter(*batch_indeces)

                # ONLY CONVERT HERE OTHERWISE GPU MEMORY IS COOKED (i think)
                # gradient tape gets too big if not batch allocated to GPU
                # also means both foward pass, backward pass and tape are all on GPU memory so relitively fast?
                #
                # if we could get tf.Dataset working, prefetch and autotune would be vey nice 
                obs = tf.concat(batch_make(self.stored_traj["observation"]), axis=0)
                action_prob_k = tf.concat(batch_make(self.stored_traj["action_prob"]), axis=0)
                adv_k = tf.concat(batch_make(self.stored_traj["adv"]), axis=0)
                action_k = tf.concat(batch_make(self.stored_traj["action"]), axis=0)
                rtg = tf.concat(batch_make(self.stored_traj["rtg"]),axis= 0)

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

        self.total_steps += (total_steps * self.d_size)
        self.total_updates = ((self.total_steps/batch_size) *epoch_num)

        self.reward_history.append(mean_reward)
        rolling_mean = np.mean(self.reward_history[-20:])
        self.step_history.append(self.total_steps)
        self.rolling_mean_history.append(rolling_mean)

        return mean_reward, total_steps

    def test(
            self,
            env,
            test_num,
    ) -> None:        
        # run test episodes
        seed = np.random.randint(0,400000,1)
        mean_reward,steps = self.__collect_data(env=env, seed = seed, step_limit=test_num, test=True)
        
        self.reward_history.append(mean_reward)
        rolling_mean = np.mean(self.reward_history[-20:])
        self.step_history.append(self.total_steps)
        self.rolling_mean_history.append(rolling_mean)
        
        print(f' =====> Test Episodes Mean Reward: {mean_reward} \n')

   
    def saveModels(
        self, 
        actor_path: str='trainedModels/null/actor_model',
        critic_path: str='trainedModels/null/critic_model',
        temp: str='placeholder',
        checkpoint: bool=False,
        saveCheckpoints: bool=False
    ) -> None:

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

        
    def loadModels(
        self, 
        actor_path: str='trainedModels/actor_model', 
        critic_path: str='trainedModels/critic_model'
    ) -> None:
        custom_objects = {"ReducedGlorot": ReducedGlorot}

        self.actor.cnn = tf.keras.models.load_model(f'{actor_path}.keras', custom_objects=custom_objects) # type: ignore
        self.critic.cnn = tf.keras.models.load_model(f'{critic_path}.keras', custom_objects=custom_objects) # type: ignore

        print(f" Models loaded from {actor_path} and {critic_path} ")
    

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
        


##################################################################################################################################################
## Private Methods
##################################################################################################################################################

    def __collect_data(
            self,
            env,
            seed,
            step_limit = 0,
            use_gae = False,
            use_adv = False,
            test = False,
        ):

        # set current lists
        t_observation, t_reward, t_terminated, t_truncated, t_info, t_action, t_action_prob, t_critic_vals = [],[],[],[],[] ,[], [], []

        # reset env, fill in rest with placeholders
        observation, info = env.reset(seed= int(seed))

        # set initial values
        observation = observation /255.0
        terminated = np.zeros(self.d_size, dtype=bool)
        truncated = np.zeros(self.d_size, dtype=bool)
        info = np.zeros(self.d_size)

        # track total
        total_reward = np.zeros(self.d_size)
        steps = 0
        episodic_reward = []

        collection_length = self.collection_size -1
        if test:
            collection_length = step_limit

        # run untill the episode ends
        for i in range(collection_length):
            action, action_prob = self.actor.choose_action(observation) # type: ignore

            t_observation.append(observation)
            t_action.append(action)
            t_action_prob.append(action_prob)
            t_terminated.append(terminated)
            t_truncated.append(truncated)
            t_info.append(info)
            t_critic_vals.append(self.critic.predict(observation).numpy().flatten())

            observation, reward, terminated, truncated, info = env.step(action)

            observation = observation /255.0

            # append reward after we observed it so S,A,R stored at the same index (makes GAE slightly easier)
            t_reward.append(reward)

            total_reward += reward

            # get terminal totals
            mask = terminated | truncated
            episodes = np.where(mask,total_reward, 12345)
            episodes = episodes[episodes != 12345]
            episodic_reward.extend(episodes)
            
            # reset envs that are terminal to 0
            total_reward = np.where(np.logical_not(mask),total_reward,0)

            steps += 1

        # append final observation
        t_observation.append(observation)
        t_action.append(action)
        t_action_prob.append(action_prob)
        t_terminated.append(terminated)
        t_truncated.append(truncated)
        t_info.append(info)
        t_reward.append(reward)
        t_critic_vals.append(self.critic.predict(observation).numpy().flatten())
        total_reward += reward
        steps += 1

        mask = (terminated | truncated)  # type: ignore
        episodes = np.where(mask,total_reward,0)
        episodes = episodes[episodes != 0]
        episodic_reward.extend(episodes)
        mean_episodic = np.mean(episodic_reward)
        
        if not test:
            # perform both rewards to go + adv as soon as trajectory done
            if use_gae:
                t_rtg, t_adv = self.__rtg_advgeneral(t_reward,t_critic_vals, t_terminated,t_truncated)
            elif use_adv:
                t_rtg, t_adv = self.__rtg_adv(t_reward,t_critic_vals)

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

        return mean_episodic, steps

#
# GAE calculation
#
    def __rtg_advgeneral(
            self,
            t_rw,
            critic_vals,
            terminated,
            truncated
        ) -> tuple:
        steps = self.collection_size

        cumulative_reward = np.zeros(self.d_size)

        # initialise rtg and advantage lists
        rewards_tg = np.zeros((steps,self.d_size),dtype=np.float32)
        gae = np.zeros((steps,self.d_size),dtype=np.float32)

        # Go back through episode to accumulate reward values
        for time in reversed(range(steps)):
            mask = np.logical_not(terminated[time] | truncated[time])

            # reward to go
            # as we store s,a,r together at the same index
            reward = t_rw[time]

            rtg = reward + (self.discount * cumulative_reward)

            rewards_tg[time] = np.where(mask,rtg,np.zeros(self.d_size))
            cumulative_reward = rewards_tg[time]

            delta = reward
            # check if not at last recorded step to avoid indexing error 
            if time < steps-1:
                # calculate td value, using same as adv
                # 𝑟𝑡 + 𝛾𝑉(𝑠𝑡+1) − 𝑉(𝑠𝑡)
                delta = reward + (self.discount * critic_vals[time + 1]) - critic_vals[time]
            
            if time< steps -1:
                gae[time] = delta + ((self.discount * self.td_lambda) * gae[time +1])
            else:
                gae[time] = delta
            
            # if terminal set to 0 otherwise keep propagating 
            gae[time] = np.where(mask,gae[time],np.zeros(self.d_size))

        return rewards_tg, gae

# Works less well than GAE but still here
    def __rtg_adv(
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
    