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
N_WORKERS = 1
N_WORKERS = multiprocessing.cpu_count()  # 并行work数量等于CPU核数
MAX_EPISODE_TIME = 10000
current_episode_time = 0
MAX_STEP_NUM = 100
ENTROPY_BETA = 0.001
LR_A = 0.001    # actor学习率
LR_C = 0.001    # critic学习率
GLOBAL_RUNNING_R = []

SCORE = []
TD_ERROR = []

env = Game()
N_S, N_A = 4, 4


class GlobalAgent:
    class __GlobalNetwork:
        def __init__(self, scope="GLOBAL_NETWORK"):
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'State')
                self.action_prob, self.estimated_value = NetworkTool.get_network_output(self.state)
                self.actor_params, self.critic_params = NetworkTool.get_network_params(scope)

    instance = None

    def __init__(self, scope="GLOBAL_NETWORK"):
        if not GlobalAgent.instance:
            GlobalAgent.instance = GlobalAgent.__GlobalNetwork(scope)

    def __getattr__(self, item):
        return getattr(self.instance, item)


class LocalAgent:
    def __init__(self, scope, global_ac=None):
        # 基础数据
        with tf.variable_scope(scope):
            # 读入State、Action和Target_Value
            self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'State')
            self.action = tf.placeholder(tf.int32, [None, 1], 'Action')
            self.target_value = tf.placeholder(tf.float32, [None, 1], 'Target_Value')

            self.action_prob, self.estimated_value = NetworkTool.get_network_output(self.state)
            self.actor_params, self.critic_params = NetworkTool.get_network_params(scope)

            self.actor_loss, self.critic_loss = \
                NetworkTool.get_loss_value(self.target_value, self.estimated_value, self.action, self.action_prob)

            self.actor_grads, self.critic_grads = \
                NetworkTool.get_loss_gradient(self.actor_loss, self.actor_params, self.critic_loss, self.critic_params)

        # 同步
        with tf.name_scope('sync'):
            # 从远程获取参数
            with tf.name_scope('pull'):
                self.pull_actor_params_op = \
                    [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, global_ac.actor_params)]
                self.pull_c_params_op = \
                    [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params, global_ac.critic_params)]
            # 向远程提交数据
            with tf.name_scope('push'):
                self.update_actor_op = OPT_A.apply_gradients(zip(self.actor_grads, global_ac.actor_params))
                self.update_critic_op = OPT_C.apply_gradients(zip(self.critic_grads, global_ac.critic_params))

    def push(self, feed_dict):
        SESS.run([self.update_actor_op, self.update_critic_op], feed_dict=feed_dict)

    def pull(self):
        SESS.run([self.pull_actor_params_op, self.pull_c_params_op])

    # 根据state选择action
    def choose_action(self, s):
        prob_weights = SESS.run(self.action_prob, feed_dict={self.state: s[np.newaxis, :]})
        # ravel()行序优先拉平
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action


class NetworkTool:

    @staticmethod
    def get_loss_value(target_value, estimated_value, action, action_prob):
        # TD Error = 目标值 - 估计值
        td_error = tf.subtract(target_value, estimated_value, name='TD_error')

        # 计算Critic loss
        with tf.name_scope('critic_loss'):
            critic_loss = tf.reduce_mean(tf.square(td_error))  # TD Error的平方和平均

        # 计算Actor loss
        # todo 重点分析Loss Function
        with tf.name_scope('actor_loss'):
            log_prob = tf.reduce_sum(tf.log(action_prob) * tf.one_hot(action, N_A, dtype=tf.float32),
                                     axis=1,
                                     keep_dims=True)
            exp_v = log_prob * td_error
            entropy = -tf.reduce_sum(action_prob * tf.log(action_prob + 1e-5),
                                     axis=1,
                                     keep_dims=True)  # encourage exploration
            exp_v = ENTROPY_BETA * entropy + exp_v
            actor_loss = tf.reduce_mean(-exp_v)

        tf.summary.scalar('actor_loss', tensor=actor_loss)
        tf.summary.scalar('critic_loss', tensor=critic_loss)

        return actor_loss, critic_loss

    # 计算Loss Function的梯度
    @staticmethod
    def get_loss_gradient(actor_loss, actor_params, critic_loss, critic_params):
        with tf.name_scope('local_gradient'):
            actor_grads = tf.gradients(actor_loss, actor_params, name='actor_gradient')
            critic_grads = tf.gradients(critic_loss, critic_params, name='critic_gradient')

        return actor_grads, critic_grads

    # Actor和Critic网络的输出
    @staticmethod
    def get_network_output(state):
        # 参数初始化工作非常重要
        w_init = tf.contrib.layers.xavier_initializer()
        state = tf.reshape(state, [-1, 4 * 4 * 1])

        with tf.variable_scope('Actor'):
            actor_fc_layer_1 = tf.layers.dense(inputs=state,
                                               units=64,
                                               activation=tf.nn.relu6,
                                               kernel_initializer=w_init,
                                               name='actor_fc_layer_1')
            actor_dropout_1 = tf.layers.dropout(
                inputs=actor_fc_layer_1, rate=0.4, name='actor_dropout_1')
            actor_fc_layer_2 = tf.layers.dense(inputs=actor_dropout_1,
                                               units=N_A,
                                               activation=tf.nn.relu,
                                               kernel_initializer=w_init,
                                               name='actor_fc_layer_2')
            action_prob = tf.nn.softmax(actor_fc_layer_2, name="action_prob")

        with tf.variable_scope('Critic'):
            critic_fc_layer_1 = tf.layers.dense(inputs=state,
                                                units=64,
                                                activation=tf.nn.relu6,
                                                kernel_initializer=w_init,
                                                name='critic_fc_layer_1')
            critic_dropout_1 = tf.layers.dropout(inputs=critic_fc_layer_1, rate=0.4)
            estimated_value = tf.layers.dense(inputs=critic_dropout_1,
                                              units=1,
                                              kernel_initializer=w_init,
                                              name='estimated_value')
            # tf.summary.scalar('estimated_value', estimated_value)

        return action_prob, estimated_value

    # Actor-Critic的参数
    @staticmethod
    def get_network_params(scope):
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Critic')

        return actor_params, critic_params


class Worker(object):
    def __init__(self, worker_name, global_agent_name):
        self.env = Game()
        self.name = worker_name
        # 所服务的全局Actor-Critic
        self.local_network = LocalAgent(worker_name, global_agent_name)

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
                action = self.local_network.choose_action(state)
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
            tf.summary.scalar('score', tf.constant(np.sum(state)))

            # 得到最后一个state的Target Value
            if is_game_over:
                target_value = 0
                print(np.sum(state))
            else:
                target_value = SESS.run(self.local_network.estimated_value,
                                        feed_dict={self.local_network.state: state[np.newaxis]})[0, 0]

            buf_target_value = Worker._get_target_value_list(reward_list=buf_reward, last_target_value=target_value)
            buf_action = np.vstack(buf_action)
            feed_dict = {
                self.local_network.state: buf_state,
                self.local_network.action: buf_action,
                self.local_network.target_value: buf_target_value
            }
            # 同步数据
            self.local_network.push(feed_dict=feed_dict)
            self.local_network.pull()

            # summary = SESS.run(merged, feed_dict={
            #     self.AC.state: buf_state[0][np.newaxis],
            #     self.AC.action: buf_action[0][np.newaxis],
            #     self.AC.target_value: buf_target_value[0][np.newaxis]
            # })
            # writer.add_summary(summary, current_episode_time)

            print("EPISODE_TIME: %d" % current_episode_time)

    # 由后往前的算Target Value
    @staticmethod
    def _get_target_value_list(reward_list, last_target_value, discount_factor=0.9):

        target_value_list = [last_target_value]

        for reward in reward_list[:-1][::-1]:
            last_target_value = reward + discount_factor * last_target_value
            target_value_list.append(last_target_value)

        target_value_list.reverse()
        return np.vstack(target_value_list)


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # Actor和Critic的学习率
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        # 输入GLOBAL_NET_SCOPE将会生成一个特殊的ACNet实例（适用于全局）
        GLOBAL_AGENT = GlobalAgent()
        # GLOBAL_AC = Agent(GLOBAL_NET_SCOPE)
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AGENT))

    # TensorFlow 用于并行的工具
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # 输出日志
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOG_DIR, SESS.graph)

    with tf.device('/gpu:0'):
        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)  # 添加一个工作线程
            t.start()
            worker_threads.append(t)

    # tf 的线程调度
    COORD.join(worker_threads)

    print(SCORE)
    print(max(SCORE))
    print(min(SCORE))
    print(sum(SCORE)/len(SCORE))
    print(TD_ERROR)

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.plot(np.arange(len(SCORE)),SCORE)
    plt.plot(np.arange(len(TD_ERROR)),TD_ERROR)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
