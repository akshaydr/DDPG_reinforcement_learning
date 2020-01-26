import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Dense, Activation, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action under a deterministic policy.
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.compat.v1.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.compat.v1.trainable_variables()[
            len(self.network_params):]

        # Operation for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
            for i in range(len(self.target_network_params))]

        # This gradient provided by the critic network
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.compat.v1.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = layers.Input(shape=[self.s_dim])
        net = Dense(400, activation='relu')(inputs)
        net = layers.BatchNormalization()(net)
        net = Dense(300, activation='relu')(net)
        net = layers.BatchNormalization()(net)

        # Final layer weights are initialized to Uniform [-0.003, 0.003]
        w_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)

        # The output layer activation is a tanh to keep the action
        # between -action_bound and action_bound
        out = Dense(self.a_dim, activation='tanh', kernel_initializer=w_init)(net)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    # Input to the network is the state and action, output is Q(s,a).

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.compat.v1.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.keras.losses.MSE(self.predicted_q_value, self.out)
        self.optimize = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.compat.v1.gradients(self.out, self.action)
        print (self.action_grads)


    def create_critic_network(self):
        # The action must be obtained from the output of the Actor network.
        inputs = layers.Input(shape=[self.s_dim])
        action = layers.Input(shape=[self.a_dim])

        net = Dense(400, activation='relu')(inputs)
        net = layers.BatchNormalization()(net)

        a_layer = Dense(400, activation='linear')(action)
        a_layer = layers.BatchNormalization()(a_layer)

        hidden = Add()([a_layer, net])
        hidden = Dense(300, activation='relu')(hidden)
        w_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)

        out = Dense(1, kernel_initializer=w_init, activation='linear')(hidden)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions })
        

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)