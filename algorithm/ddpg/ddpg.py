# -*- coding: utf-8 -*-

from algorithm.ddpg.actor import *
from algorithm.ddpg.critic import *
from algorithm.ddpg.replay import *

import numpy as np

GAMMA = 0.99


def ddpg(sess, game_env, max_episode_num=2000, max_step_num=1000):

    # Initialize Estimate and Target Network, initialize Replay Buffer
    actor, critic, replay = Actor(sess, game_env), Critic(sess, game_env), Replay()
    # Data Analysis
    score_list = []

    total_step_num = 0
    current_episode_num = 0
    while current_episode_num < max_episode_num:
        state = game_env.reset()

        current_step_num = 0
        while current_step_num < max_step_num:
            # Select Action according to State
            action = actor.get_action(np.array(state), 'Estimate')[0]
            # Execute this action and get feedback
            next_state, reward, done = game_env.step(action)
            # Store the transition to Replay Buffer
            replay.store([state, action, reward, next_state])

            if done or replay.filled():
                score_list.append(np.sum(state))
                print("Current Episode num is %s ..." % current_episode_num)
                sample_replay = replay.sample()
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
                              t_q_v=target_q_value,
                              iter_num=total_step_num)

                actor.update('Estimate',
                             a=sample_replay['action'],
                             s=sample_replay['state'],
                             e_q_v=estimate_q_value,
                             iter_num=total_step_num)

                critic.update('Target')
                actor.update('Target')

                break

            state = next_state
            current_step_num += 1
            total_step_num += 1

        current_episode_num += 1

    return score_list
