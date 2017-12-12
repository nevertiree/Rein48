# -*- coding: utf-8 -*-

import multiprocessing  # 用于统计CPU核数
import threading
import os
import shutil

from game.game_cli import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_GRAPH = True
LOG_DIR = './log'  # 输出log目录
N_WORKERS = multiprocessing.cpu_count()  # 并行work数量等于CPU核数
MAX_EPISODE_TIME = 10000
GLOBAL_NET_SCOPE = 'Global_Net'  # 是否为全局网络
current_episode_time = 0
MAX_STEP_NUM = 100
GAMMA = 0.9  # 折扣率
ENTROPY_BETA = 0.001
LR_A = 0.001    # actor学习率
LR_C = 0.001    # critic学习率
GLOBAL_RUNNING_R = []

SCORE = []
TD_ERROR = []

env = Game()
N_S = 4
N_A = 4


class Agent(object):
    def __init__(self, scope, global_ac=None):
        # Agent分为两种：全局Agent负责汇总数据，本地Agent负责计算loss

        # 全局Agent
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                # 读入State，此处为 4 * 4 的矩阵
                self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'State')

                # 取build_net()的最后两个返回值
                # 前两个返回值是action_prob和estimated_value，只有本地Agent需要
                self.actor_params, self.critic_params = self._get_network_output(scope)[-2:]
        # 本地Agent
        else:
            # 基础数据
            with tf.variable_scope(scope):
                # 读入State、Action和Target_Value
                self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'State')
                self.action = tf.placeholder(tf.int32, [None, ], 'Action')
                self.target_value = tf.placeholder(tf.float32, [None, 1], 'Target_Value')

                # 取build_net()全部返回值
                self.action_prob, self.estimated_value, self.actor_params, self.critic_params = self._get_network_output(scope)

                self.actor_loss, self.critic_loss = \
                    Agent._get_loss_value(self.target_value, self.estimated_value, self.action, self.action_prob)

                self.actor_grads, self.critic_grads = \
                    Agent._get_loss_gradient(self.actor_loss, self.actor_params, self.critic_loss, self.critic_params)

            # 同步
            with tf.name_scope('sync'):
                # 从远程获取参数
                with tf.name_scope('pull'):
                    self.pull_actor_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.actor_params, global_ac.actor_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.critic_params, global_ac.critic_params)]
                # 向远程提交数据
                with tf.name_scope('push'):
                    self.update_actor_op = OPT_A.apply_gradients(zip(self.actor_grads, global_ac.actor_params))
                    self.update_critic_op = OPT_C.apply_gradients(zip(self.critic_grads, global_ac.critic_params))

    @staticmethod
    def _get_loss_value(target_value, estimated_value, action, action_prob):
        # 根据TD Error计算Loss
        td_error = tf.subtract(target_value,estimated_value, name='TD_error')

        # 计算Critic loss
        with tf.name_scope('c_loss'):
            # TD Error的平方和平均
            critic_loss = tf.reduce_mean(tf.square(td_error))

        # 计算Actor loss
        # todo 重点分析Loss Function
        with tf.name_scope('a_loss'):
            log_prob = tf.reduce_sum(tf.log(action_prob) * tf.one_hot(action, N_A, dtype=tf.float32),
                                     axis=1,
                                     keep_dims=True)
            exp_v = log_prob * td_error
            entropy = -tf.reduce_sum(action_prob * tf.log(action_prob + 1e-5),
                                     axis=1,
                                     keep_dims=True)  # encourage exploration
            exp_v = ENTROPY_BETA * entropy + exp_v
            actor_loss = tf.reduce_mean(-exp_v)

        return actor_loss, critic_loss

    # 计算Loss Function的梯度
    @staticmethod
    def _get_loss_gradient(actor_loss, actor_params, critic_loss, critic_params):
        with tf.name_scope('local_grad'):
            actor_grads = tf.gradients(actor_loss, actor_params)
            critic_grads = tf.gradients(critic_loss, critic_params)
        return actor_grads, critic_grads

    # 搭建Actor和Critic网络
    def _get_network_output(self, scope):
        w_init = tf.random_normal_initializer(0., .1)

        # Actor网络
        with tf.variable_scope('actor'):
            # todo 卷积神经网络设计
            flat_state = tf.reshape(self.state, [-1, 4 * 4 * 1])
            flat = tf.layers.dense(inputs=flat_state,
                                   units=64,
                                   activation=tf.nn.relu6,
                                   kernel_initializer=w_init,
                                   name='la')
            dense = tf.layers.dense(inputs=flat,
                                    units=N_A,
                                    activation=tf.nn.softmax,
                                    kernel_initializer=w_init,
                                    name='ap')
            action_prob = tf.nn.softmax(dense, name="softmax_tensor")

        # Critic网络
        # todo 卷积神经网络设计
        with tf.variable_scope('critic'):
            flat_state = tf.reshape(self.state, [-1, 4 * 4 * 1])
            l_c = tf.layers.dense(inputs=flat_state,
                                  units=64,
                                  activation=tf.nn.relu6,
                                  kernel_initializer=w_init,
                                  name='lc')
            estimated_value = tf.layers.dense(inputs=l_c,
                                              units=1,
                                              kernel_initializer=w_init,
                                              name='estimated_value')

        # Actor-Critic的参数
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return action_prob, estimated_value, actor_params, critic_params

    def update_global(self, feed_dict):
        SESS.run([self.update_actor_op, self.update_critic_op], feed_dict=feed_dict)

    def pull_global(self):
        SESS.run([self.pull_actor_params_op, self.pull_c_params_op])

    # 根据state选择action
    def choose_action(self, s):
        prob_weights = SESS.run(self.action_prob, feed_dict={self.state: s[np.newaxis, :]})
        # ravel()行序优先拉平
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, global_ac):
        self.env = Game()
        self.name = name
        # 所服务的全局Actor-Critic
        self.AC = Agent(name, global_ac)

    def work(self):
        global GLOBAL_RUNNING_R, current_episode_time
        current_episode_time = 0

        # 整个训练在迭代MAX_EPISODE_TIME次后结束
        while not COORD.should_stop() and current_episode_time < MAX_EPISODE_TIME:
            # 新开一个episode, reward清零
            state, is_game_over = self.env.reset(), False
            state = np.array(state)
            current_step_num = 0
            # 缓存(用于n_steps更新)
            buf_state, buf_action, buf_reward = [], [], []

            # 等到Episode结束或者经过较长的一段间隔
            while not is_game_over and current_step_num < MAX_STEP_NUM:
                # 根据state做出action，并得到new state
                action = self.AC.choose_action(state)
                state, reward, is_game_over = self.env.step(action)
                state = np.array(state)
                # 存储序列值
                buf_state.append(state)
                buf_action.append(action)
                buf_reward.append(reward)
                # 迭代计数控制器
                current_step_num += 1
                current_episode_time += 1

            SCORE.append(np.sum(state))

            # 由后往前的算Target Value
            if is_game_over:
                target_value = 0
                print(np.sum(state))
            else:
                target_value = SESS.run(self.AC.estimated_value, {self.AC.state: state[np.newaxis]})[0, 0]

            # Forward View计算各个State的Target Value
            buf_target_value = []
            for reward in buf_reward[::-1]:
                target_value = reward + GAMMA * target_value
                buf_target_value.append(target_value)
            buf_target_value.reverse()

            # 数据：用于更新Actor和Critic的参数
            buf_target_value = np.vstack(buf_target_value)

            feed_dict = {
                self.AC.state: buf_state,
                self.AC.action: buf_action,
                self.AC.target_value: buf_target_value,
            }
            # print(feed_dict)

            buf_state, buf_action, buf_reward = [], [], []

            # 同步数据
            self.AC.update_global(feed_dict)
            self.AC.pull_global()

            print("EPISODE_TIME: %d" % current_episode_time)


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # Actor和Critic的学习率
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        # 输入GLOBAL_NET_SCOPE将会生成一个特殊的ACNet实例（适用于全局）
        GLOBAL_AC = Agent(GLOBAL_NET_SCOPE)
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    # TensorFlow 用于并行的工具
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # 输出日志
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)  # 添加一个工作线程
        t.start()
        worker_threads.append(t)
    # tf 的线程调度
    COORD.join(worker_threads)

    print(SCORE)
    print(TD_ERROR)

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.plot(np.arange(len(SCORE)),SCORE)
    plt.plot(np.arange(len(TD_ERROR)),TD_ERROR)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
