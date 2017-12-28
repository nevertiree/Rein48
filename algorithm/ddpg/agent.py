# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class Agent:

    def __init__(self, game_env, tau=0.9):
        self.tau = tau

        self.action_size = game_env.action_size
        self.state_size = game_env.state_size
        self.reward_size = game_env.reward_size

        self.next_state = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.state_size, self.state_size],
                                         name='Next_State')

        self.state = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.state_size, self.state_size],
                                    name='State')

        self.action = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.action_size],
                                     name='Action')

        self.reward = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.reward_size],
                                     name='Reward')

        self.q_value = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.reward_size],
                                      name='Q_Value')

    @staticmethod
    def num_2_one_hot(value, max_value):
        one_hot_array = np.zeros(max_value)
        one_hot_array[value] = 1
        return one_hot_array
