# -*- coding: utf-8 -*-

import os
import shutil

from algorithm.ddpg.agent import *

import tensorflow as tf
import numpy as np


class Actor(Agent):

    def __init__(self, sess, game_env):
        super(Actor, self).__init__(game_env)
        self.sess = sess
        self.log_dir = 'log/actor'

        with tf.variable_scope('Actor'):
            self.estimate_action = self.actor_network()
            self.estimate_para = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope='Actor/Network')

        with tf.variable_scope('Target_Actor'):
            self.target_action = self.actor_network()
            self.target_para = self.estimate_para

        with tf.variable_scope('Actor/Update'):
            self.optimizer = tf.train.RMSPropOptimizer(0.001)
            self.action_gradient = tf.gradients(ys=self.q_value,
                                                xs=self.estimate_action)
            self.policy_gradient = tf.gradients(ys=self.estimate_action,
                                                xs=self.estimate_para,
                                                grad_ys=self.action_gradient)
            self.estimate_update = self.optimizer.apply_gradients(grads_and_vars=
                                                                  zip(self.policy_gradient, self.estimate_para))

        with tf.variable_scope('Target_Actor/Update'):
            self.target_update = [tf.assign(ref=t, value=self.tau * t + (1 - self.tau) * e)
                                  for t, e in zip(self.target_para, self.estimate_para)]

        with tf.variable_scope("A_Summary"):
            self.merge = tf.summary.merge_all()
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            self.train_writer = tf.summary.FileWriter(logdir=self.log_dir + '/train', graph=self.sess.graph)
            # self.test_writer = tf.summary.FileWriter(logdir=self.logdir + '/test', graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def actor_network(self):

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            init_w = tf.random_normal_initializer(mean=1, stddev=2)
            init_b = tf.constant_initializer(value=1)

            input_layer = tf.reshape(tensor=self.state,
                                     shape=[-1, self.state_size, self.state_size, 1],
                                     name='Input_Layer')

            conv_layer_1 = tf.layers.conv2d(inputs=input_layer,
                                            filters=32,
                                            kernel_size=[2, 2],
                                            padding='valid',
                                            kernel_initializer=init_w,
                                            bias_initializer=init_b,
                                            activation=tf.nn.relu)

            conv_layer_2 = tf.layers.conv2d(inputs=conv_layer_1,
                                            filters=64,
                                            kernel_size=[2, 2],
                                            padding='valid',
                                            kernel_initializer=init_w,
                                            bias_initializer=init_b,
                                            activation=tf.nn.relu)

            reshape = tf.reshape(tensor=conv_layer_2, shape=[-1, 2 * 2 * 64])

            output = tf.layers.dense(inputs=reshape,
                                     units=self.action_size,
                                     activation=tf.nn.softmax,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b)

            return output

    def get_action(self, s, network_type):
        s = s[np.newaxis]
        if len(s.shape) > 3:
            s = np.squeeze(s)
        if network_type == 'Target':
            action_prob_list = self.sess.run(self.target_action, feed_dict={self.state: s})
        else:
            action_prob_list = self.sess.run(self.estimate_action, feed_dict={self.state: s})
        action = [np.random.choice(a=range(action_prob.shape[0]), p=action_prob.ravel())
                  for action_prob in action_prob_list]

        # tf.summary.histogram(name='Action', values=action)
        return action

    def update(self, network_type, e_q_v=None, a=None, s=None, iter_num=None):
        if network_type == 'Estimate':
            a = [Actor.num_2_one_hot(a_item, self.action_size) for a_item in a]
            self.sess.run(self.estimate_update, feed_dict={
                self.estimate_action: a,
                self.state: s,
                self.q_value: e_q_v
            })
        else:
            self.sess.run(self.target_update)
