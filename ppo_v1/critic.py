import gymnasium as gym
import tensorflow as tf
from model.cnn import define_model

CLIPNORM = 0.5

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

    # @tf.function
    def train(
        self, 
        optimiser,
        obs,
        rtg
    ) -> None:

        # SAME AGAIN GPU COOKED 
        obs = tf.concat(obs, axis=0)
        # dimension added on the last one just to match the value format, idk if this actually fixes anything
        rtg = tf.expand_dims(tf.convert_to_tensor(rtg),axis=-1)

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
        
        grad_clipped, global_norm = tf.clip_by_global_norm(gradients, CLIPNORM)

        optimiser.apply_gradients(zip(grad_clipped, self.cnn.trainable_variables)) # type: ignore
