# -*- coding: utf-8 -*-

from algorithm.ddpg.agent import *

import tensorflow as tf
import numpy as np


class Critic(Agent):

    def __init__(self, sess, game_env):
        super(Critic, self).__init__(game_env)
        self.sess = sess
        with tf.variable_scope('Estimate_Critic'):
            self.estimate_q_value = self.critic_network('Estimate')
            self.estimate_para = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope='Estimate_Critic/Network')

        with tf.variable_scope('Target_Critic'):
            self.target_q_value = self.critic_network('Target')

            self.target_para = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope='Target_Critic/Network')

        with tf.variable_scope('Critic_Update'):
            self.loss = tf.reduce_mean(tf.squared_difference(x=self.estimate_q_value,
                                                             y=self.target_q_value))
            self.optimizer = tf.train.AdamOptimizer()
            self.estimate_update = self.optimizer.minimize(self.loss)
            self.target_update = [tf.assign(ref=t, value=self.tau * t + (1 - self.tau) * e)
                                  for t, e in zip(self.target_para, self.estimate_para)]

        self.sess.run(tf.global_variables_initializer())

    def critic_network(self, network_type):
        if network_type == 'Estimate':
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(network_type + '_Critic/Network'):
            init_w = tf.random_normal_initializer(mean=1, stddev=5)
            init_b = tf.constant_initializer(1)

            layer_unit = 32

            state_weight = tf.get_variable(name='state_weight',
                                           shape=[self.state_size * self.state_size, layer_unit],
                                           initializer=init_w,
                                           trainable=trainable)

            action_weight = tf.get_variable(name='action_weight',
                                            shape=[self.action_size, layer_unit],
                                            initializer=init_w,
                                            trainable=trainable)
            bias = tf.get_variable(name='bias', shape=[1, layer_unit], trainable=trainable)

            reshape_state = tf.reshape(tensor=self.state, shape=[-1, self.state_size * self.state_size])
            concat = tf.nn.relu(tf.matmul(reshape_state, state_weight) +
                                tf.matmul(self.action, action_weight) +
                                bias)

            output = tf.layers.dense(inputs=concat,
                                     units=1,
                                     activation=tf.nn.softmax,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     trainable=trainable)

            return output

    def get_q_value(self, a, s, network_type):
        a = [Critic.num_2_one_hot(a_item, self.action_size) for a_item in a]
        s = s[:np.newaxis]

        if network_type == 'Estimate':
            return self.sess.run(self.estimate_q_value, feed_dict={self.action: a, self.state: s})
        else:
            return self.sess.run(self.target_q_value, feed_dict={self.action: a, self.state: s})

    def update(self, network_type, s=None, a=None, t_q_v=None):
        if network_type == 'Estimate':
            a = [Critic.num_2_one_hot(a_item, self.action_size) for a_item in a]
            self.sess.run(self.estimate_update, feed_dict={
                self.state: s,
                self.action: a,
                self.target_q_value: t_q_v
            })
        else:
            self.sess.run(self.target_update)

    @staticmethod
    def num_2_one_hot(value, max_value):
        one_hot_array = np.zeros(max_value)
        one_hot_array[value] = 1
        return one_hot_array
