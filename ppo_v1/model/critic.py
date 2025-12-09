import gymnasium as gym
import tensorflow as tf
from model.cnn import define_model
from main import GRADNORM

##
## Critic network
##
class Critic:    
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
    
    def predict(self, state):
        # value = tf.reshape((self.cnn(state)),[-1]).numpy()
        value = self.cnn(state)
        return value

    @tf.function
    def train(
        self, 
        optimiser,
        obs,
        rtg,
    ) -> None:

        with tf.GradientTape() as tape:
            value = self.predict(obs)
            diff = (value - rtg)

            # print(obs)
            # print(value)
            # print(rtg)
            # print(diff)

            loss = tf.square(diff)

            loss = tf.reduce_mean(loss)

            # print(f"Critic Loss: {loss}")

        gradients = tape.gradient(loss, self.cnn.trainable_variables)
        
        grad_clipped, global_norm = tf.clip_by_global_norm(gradients, GRADNORM)

        optimiser.apply_gradients(zip(grad_clipped, self.cnn.trainable_variables)) # type: ignore
