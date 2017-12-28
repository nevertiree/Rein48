# -*- coding: utf-8 -*-

import os
import shutil

from algorithm.ddpg.agent import *

import tensorflow as tf
import numpy as np


class Critic(Agent):

    def __init__(self, sess, game_env):
        super(Critic, self).__init__(game_env)
        self.sess = sess
        self.log_dir = 'log/critic'

        with tf.variable_scope('Critic'):
            self.estimate_q_value = self.critic_network('Estimate')
            self.estimate_para = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope='Critic/Network')

        with tf.variable_scope('Target_Critic'):
            self.target_q_value = self.critic_network('Target')

            self.target_para = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope='Target_Critic/Network')

        with tf.variable_scope('Critic/Update'):
            self.loss = tf.reduce_mean(tf.squared_difference(x=self.estimate_q_value,
                                                             y=self.target_q_value), name='TD_Error')
            tf.summary.scalar(name='TD_Error', tensor=self.loss)
            self.optimizer = tf.train.AdamOptimizer()
            self.estimate_update = self.optimizer.minimize(self.loss)

        with tf.variable_scope("Target_Critic/Update"):
            self.target_update = [tf.assign(ref=t, value=self.tau * t + (1 - self.tau) * e)
                                  for t, e in zip(self.target_para, self.estimate_para)]

        with tf.variable_scope("C_Summary"):
            self.merge = tf.summary.merge_all()
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            self.train_writer = tf.summary.FileWriter(logdir=self.log_dir + '/train', graph=self.sess.graph)
            # self.test_writer = tf.summary.FileWriter(logdir=self.logdir + '/test', graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def critic_network(self, network_type):
        if network_type == 'Estimate':
            trainable = True
        else:
            trainable = False

        with tf.variable_scope('Network'):
            init_w = tf.random_normal_initializer(mean=1, stddev=5)
            init_b = tf.constant_initializer(1)

            layer_unit = 32

            state_weight = tf.get_variable(name='State_Weight',
                                           shape=[self.state_size * self.state_size, layer_unit],
                                           initializer=init_w,
                                           trainable=trainable)

            action_weight = tf.get_variable(name='Action_Weight',
                                            shape=[self.action_size, layer_unit],
                                            initializer=init_w,
                                            trainable=trainable)
            bias = tf.get_variable(name='Bias', shape=[1, layer_unit], trainable=trainable)

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
            tf.summary.histogram(name='Q_Value', values=output)
            return output

    def get_q_value(self, a, s, network_type):
        a = [Critic.num_2_one_hot(a_item, self.action_size) for a_item in a]
        s = s[:np.newaxis]

        if network_type == 'Estimate':
            return self.sess.run(self.estimate_q_value, feed_dict={self.action: a, self.state: s})
        else:
            return self.sess.run(self.target_q_value, feed_dict={self.action: a, self.state: s})

    def update(self, network_type, s=None, a=None, t_q_v=None, iter_num=None):
        if network_type == 'Estimate':
            a = [Critic.num_2_one_hot(a_item, self.action_size) for a_item in a]
            summary, _ = self.sess.run([self.merge, self.estimate_update], feed_dict={
                self.state: s,
                self.action: a,
                self.target_q_value: t_q_v
            })
            self.train_writer.add_summary(summary, iter_num)
        else:
            self.sess.run(self.target_update)
