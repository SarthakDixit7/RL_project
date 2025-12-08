import tensorflow as tf
import numpy as np
from model.cnn import define_model

CLIPNORM = 0.5

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
        ) -> None:

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
    # the action prob is used inside the gpu tf portions so thats fine to return a tf tensor
    def choose_action(self, state) -> tuple:
        
        probabilities = tf.reshape((self.cnn(state)),[-1]).numpy()

        action = np.random.choice(len(probabilities), p=probabilities)

        probability = probabilities[action]

        return (action, probability) # type: ignore

    def give_action_prob(self, state):

        probabilities = self.cnn(state)
        return probabilities
    
    # @tf.function
    def train(
        self,
        optimiser,
        obs, 
        action_prob_k, 
        adv_k,
        action_k,
        eps
    ) -> None:

        # ONLY CONVERT HERE OTHERWISE GPU MEMORY IS COOKED - gradient tape gets too big if not batch allocated to GPU
        # again expanding dims to match the concatenated observations, idk if this fixes calc problems but at least its consistent
        # i.e 
        # from 
        # tf.Tensor(x1,x2, ...], shape=(35,), dtype=float32)
        # to
        # tf.Tensor([],[], ...], shape=(35, 1), dtype=float32)
        obs = tf.concat(obs, axis=0)
        action_prob_k = tf.expand_dims(tf.convert_to_tensor(action_prob_k), axis=-1)
        adv_k = tf.expand_dims(tf.convert_to_tensor(adv_k), axis=-1)
        action_k = tf.expand_dims(tf.convert_to_tensor(action_k), axis=-1)

        # print(obs)
        # print(action_prob_k)
        # print(adv_k)
        # print(action_k)

        with tf.GradientTape() as tape:

            action_prob_current = self.give_action_prob(obs)

            # takes in probability array, creates compressed array only with the new action prob of action chosen by k
            # the batching just tells it which axis to use (i think)
            action_prob_current = tf.gather(action_prob_current,action_k, batch_dims=1)

            # clip using tensorflow to try and speed up this monstrosity
            clip = tf.where(adv_k>=0,(1 + eps) * adv_k, (1 - eps) * adv_k )

            x = tf.where(action_prob_k >0, action_prob_current / action_prob_k, 0)

            imp_s = tf.multiply(adv_k , x)

            # minus cuz idk how to maximise
            # side note you need to use miltiply, just putting -1 in front was a very painful bug
            loss = tf.multiply(tf.minimum(imp_s , clip),-1)

            loss = tf.reduce_mean(loss)

            # print(f"Actor Loss: {loss}")

        gradients = tape.gradient(loss, self.cnn.trainable_variables)

        grad_clipped, global_norm = tf.clip_by_global_norm(gradients, CLIPNORM)

        optimiser.apply_gradients(zip(grad_clipped, self.cnn.trainable_variables)) # type: ignore
