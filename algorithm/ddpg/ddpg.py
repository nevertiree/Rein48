# -*- coding: utf-8 -*-

import os
import time
import shutil

from algorithm.ddpg.actor import *
from algorithm.ddpg.critic import *
from algorithm.ddpg.replay import *
from game.game_cli import *

import tensorflow as tf
import numpy as np

MAX_EPISODE_NUM = 10
MAX_STEP_NUM = 100

GAMMA = 0.9


def run(sess, game_env):

    # Initialize Estimate and Target Network
    actor, critic = Actor(sess, game_env), Critic(sess, game_env)
    # Initialize Replay Buffer
    replay = Replay()

    current_episode_num = 0
    while current_episode_num < MAX_EPISODE_NUM:
        # do something
        state = game_env.reset()

        current_step_num = 0
        while current_step_num < MAX_STEP_NUM:
            # Select Action according to State
            action = actor.get_action(np.array(state), 'Estimate')[0]
            # Execute this action and get feedback
            next_state, reward, done = game_env.step(action)
            # Store the transition to Replay Buffer
            replay.store([state, action, reward, next_state])

            if done or replay.filled():
                sample_replay = replay.sample()
                # print(sample_replay['next_state'])
                target_next_action = actor.get_action(s=sample_replay['next_state'],
                                                      network_type='Target')
                target_next_q_value = critic.get_q_value(a=target_next_action,
                                                         s=sample_replay['next_state'],
                                                         network_type='Target')
                target_q_value = reward + GAMMA * target_next_q_value
                estimate_q_value = critic.get_q_value(a=sample_replay['action'],
                                                      s=sample_replay['state'],
                                                      network_type='Estimate')

                critic.update('Estimate',
                              a=sample_replay['action'],
                              s=sample_replay['state'],
                              t_q_v=target_q_value)

                actor.update('Estimate',
                             a=sample_replay['action'],
                             s=sample_replay['state'],
                             e_q_v=estimate_q_value)

                critic.update('Target')
                actor.update('Target')

                break

            state = next_state
            current_step_num += 1

        current_episode_num += 1


if __name__ == '__main__':
    game_environment = Game()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # actor_inst, critic_inst = Actor(session, game_environment), Critic(session, game_environment)
        run(session, game_environment)
