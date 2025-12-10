import tensorflow as tf
import numpy as np
from model.cnn import define_model

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
        
        probabilities = tf.reshape((self.cnn(state)),[-1]).numpy()

        action = np.random.choice(len(probabilities), p=probabilities)

        probability = probabilities[action]

        return (action, probability) # type: ignore

    def give_action_prob(self, state):

        probabilities = self.cnn(state)
        return probabilities
    
    def get_log_probs(self, state, actions):
        
        action_probs = self.give_action_prob(state)
        
        action_probs = tf.clip_by_value(action_probs, 1e-8, 1.0)
        
        selected_probs = tf.gather(action_probs, indices=actions, batch_dims=1)
        
        return tf.math.log(selected_probs)
    
    def apply_gradients(self, tape, loss, optimiser):
        
        gradients = tape.gradient(loss, self.cnn.trainable_variables)
        
        clipped_grads, _ = tf.clip_by_global_norm(gradients, self.gradnorm)
        
        optimiser.apply_gradients(zip(clipped_grads, self.cnn.trainable_variables))
        
    #https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9520424
    def compute_actor_loss(
        self,
        observations,
        actions,
        advantages,
        old_log_probs,
        epsilon,
        kl_coef,
        include_entropy: bool = False,
    ):
        new_log_probs = self.get_log_probs(observations, actions)
        ratio = tf.exp(new_log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
        policy_loss = tf.minimum(ratio * advantages, clipped_ratio * advantages)
        
        if include_entropy:
            action_probs = self.give_action_prob(observations)
            entropy_term = tf.reduce_sum(-action_probs * tf.math.log(action_probs + 1e-8), axis=1, keepdims=True)
            policy_loss += self.entropy * entropy_term
        
        loss = -tf.reduce_mean(policy_loss)
        kl = tf.reduce_mean(old_log_probs - new_log_probs)
        
        return loss + kl_coef * kl, kl
    
    @tf.function
    def train(
        self,
        optimiser,
        obs, 
        action_prob_k, 
        adv_k,
        action_k,
        include_entropy,
        epsilon,
        kl_coef,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # DONT ADD ANYTHING NOT TENSORFLOW HERE, otherwise tape gets all weird i think?
        with tf.GradientTape() as tape:
            
            _, kl = self.compute_actor_loss(
                observations=obs,
                actions=action_k,
                advantages=adv_k,
                old_log_probs=tf.math.log(action_prob_k + 1e-8),
                epsilon=epsilon,
                kl_coef=kl_coef,
                include_entropy=include_entropy,
            )

            # returns a tensor of probabilities (batch_size , num_actions)
            action_probs = self.give_action_prob(obs)

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

            #if include_entropy:
            #    entropy_term = tf.multiply(action_probs, tf.multiply(tf.math.log(action_probs),-1))
            #    weighted_entropy = tf.multiply(entropy_term, self.entropy)
            #    loss = tf.add( loss , weighted_entropy)

            # minus cuz idk how to maximise
            # side note you need to use miltiply, just putting -1 in front was a very painful bug
            #loss = tf.reduce_mean(tf.multiply(loss,-1))

            # print(f"Actor Loss: {loss}")

        gradients = tape.gradient(loss, self.cnn.trainable_variables)

        # couldnt get clip by norm to work which is supposed to be faster 
        grad_clipped, _ = tf.clip_by_global_norm(gradients, self.gradnorm) # type: ignore

        optimiser.apply_gradients(zip(grad_clipped, self.cnn.trainable_variables)) # type: ignore
        
        return loss, kl
