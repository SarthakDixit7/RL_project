import tensorflow as tf
import numpy as np
from ppo_v1.model.cnn import define_model

##
## Actor network
##
class Actor:    
    def __init__(
            self, 
            dimensions,
            regression,
            total_moves,
            conv,
            conv_filters,
            dense_units,
            eps_clip,
            gradnorm,
            entropy,
        ) -> None:

        self.eps = eps_clip
        self.gradnorm = gradnorm
        self.entropy = entropy

        # CNN model
        self.cnn = define_model(
            dimensions = dimensions, 
            regression=regression, 
            total_moves=total_moves, 
            conv=conv,
            conv_filters=conv_filters,
            dense_units=dense_units
        )

    # Note choose action uses numpy as thats whats used for collection
    def choose_action(self, state) -> tuple:
        
        probabilities = self.cnn(state).numpy()

        games, action_num = probabilities.shape

        # The list has to be this
        actions = np.zeros(games, dtype = np.int32)
        probs_out = []

        for game in range(games):
            actions[game] =  np.random.choice(action_num, p=probabilities[game,:])
            probs_out.append(probabilities[game,actions[game]])
        
        return (actions, probs_out) # type: ignore
    
    def choose_training_action(self, state) -> tuple:
        
        # remove extra tensor dimension as this is only used when playing
        # flatten into numpy to stop weird tf numpy stuff
        probabilities = tf.reshape((self.cnn(state)),[-1]).numpy()

        # pick 
        action = np.random.choice(len(probabilities), p=probabilities)

        probability = probabilities[action]

        return (action, probability) # type: ignore

    def give_action_prob(self, state):

        probabilities = self.cnn(state)
        return probabilities
    
    @tf.function
    def train(
        self,
        optimiser,
        obs, 
        action_prob_k, 
        adv_k,
        action_k,
        include_entropy
    ) -> None:
        # DONT ADD ANYTHING NOT TENSORFLOW HERE, otherwise tape gets all weird i think?
        with tf.GradientTape() as tape:

            # returns a tensor of probabilities (batch_size , num_actions)
            action_probs = self.give_action_prob(obs)
            print()

            # takes in probability tensor above, creates new tensor with only the prob of the selected action 
            # so just probabilities[action] but for each probability vector (only of y put dims 1)
            action_prob_current = tf.gather(action_probs, indices = action_k, batch_dims=1)

            # clip using tensorflow to try and speed up this monstrosity
            # cant use tf.cond elementwise but this basically uses a mask
            # https://stackoverflow.com/questions/37912161/how-can-i-compute-element-wise-conditionals-on-batches-in-tensorflow
            clip = tf.where(adv_k>=0,(1 + self.eps) * adv_k, (1 - self.eps) * adv_k )

            x = tf.where(action_prob_k >0, action_prob_current / action_prob_k, 0)

            imp_s = tf.multiply(adv_k , x)

            loss = tf.minimum(imp_s , clip)

            if include_entropy:
                entropy_term = tf.reduce_sum(tf.multiply(action_probs, tf.multiply(tf.math.log(action_probs),-1)),1)
                weighted_entropy = tf.multiply(entropy_term, self.entropy)
                loss = tf.add( loss , weighted_entropy)

            # minus cuz idk how to maximise
            # side note you need to use miltiply, just putting -1 in front was a very painful bug
            loss = tf.reduce_mean(tf.multiply(loss,-1))

            # print(f"Actor Loss: {loss}")

        gradients = tape.gradient(loss, self.cnn.trainable_variables)

        # couldnt get clip by norm to work which is supposed to be faster 
        grad_clipped, _ = tf.clip_by_global_norm(gradients, self.gradnorm) # type: ignore

        optimiser.apply_gradients(zip(grad_clipped, self.cnn.trainable_variables)) # type: ignore
