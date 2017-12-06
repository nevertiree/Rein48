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
MAX_GLOBAL_EP = 10000
GLOBAL_NET_SCOPE = 'Global_Net'  # 是否为全局网络
UPDATE_GLOBAL_ITER = 10  # 更新Global的频率
GAMMA = 0.9  # 折扣率
ENTROPY_BETA = 0.001
LR_A = 0.001    # actor学习率
LR_C = 0.001    # critic学习率
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

SCORE = []

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
                self.actor_params, self.critic_params = self._build_net(scope)[-2:]
        # 本地Agent
        else:
            # 基础数据
            with tf.variable_scope(scope):
                # 读入State、Action和Target_Value
                self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'State')
                self.action = tf.placeholder(tf.int32, [None, ], 'Action')
                self.target_value = tf.placeholder(tf.float32, [None, 1], 'Target_Value')

                # 取build_net()全部返回值
                self.action_prob, self.estimated_value, self.actor_params, self.critic_params = self._build_net(scope)

                # 根据TD Error计算Loss
                td_error = tf.subtract(self.target_value, self.estimated_value, name='TD_error')

                # 计算Critic loss
                with tf.name_scope('c_loss'):
                    # TD Error的平方和平均
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                # 计算Actor loss
                # todo 重点分析Loss Function
                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.action_prob) * tf.one_hot(self.action, N_A, dtype=tf.float32),
                                             axis=1,
                                             keep_dims=True)
                    exp_v = log_prob * td_error
                    entropy = -tf.reduce_sum(self.action_prob * tf.log(self.action_prob + 1e-5),
                                             axis=1,
                                             keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.actor_loss = tf.reduce_mean(-self.exp_v)

                # 计算Loss Function的梯度
                with tf.name_scope('local_grad'):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

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

    # 搭建Actor和Critic网络
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)

        # Actor网络
        with tf.variable_scope('actor'):
            # todo 卷积神经网络设计
            flat_state = tf.reshape(self.state, [-1, 4 * 4 * 1])
            flat = tf.layers.dense(inputs=flat_state,
                                   units=1024,
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
                                  units=100,
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
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        # 缓存 用于n_steps更新
        buf_state, buf_action, buf_reward = [], [], []

        # GLOBAL_EP是迭代次数
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # 新开一个episode, reward清零
            state = np.array(self.env.reset())
            ep_r = 0
            # 迭代直至episode结束
            while True:
                # 根据state做出action，并得到new state
                action = self.AC.choose_action(state)
                new_state, reward, done = self.env.step(action)
                new_state = np.array(new_state)

                # Done说明本episode结束
                if done:
                    reward = -5

                # 更新总的reward值
                ep_r += reward

                # 把state action reward 做一次记录，每个Iter清理一次
                buf_state.append(state)
                buf_action.append(action)
                buf_reward.append(reward)

                # 每 UPDATE_GLOBAL_ITER 步 或者回合完了, 进行 sync 操作
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    # 获得用于计算 TD error的下一state的value
                    if done:
                        new_state_value = 0
                        this_score = np.sum(new_state)
                        SCORE.append(this_score)
                        print(this_score)
                    else:
                        new_state_value = SESS.run(self.AC.estimated_value,
                                                   {self.AC.state: new_state[np.newaxis, :]})[0, 0]

                    # 用n_steps forward view计算Target state value
                    buf_target_value = []
                    # reverse buffer r
                    for reward in buf_reward[::-1]:
                        new_state_value = reward + GAMMA * new_state_value
                        buf_target_value.append(new_state_value)
                    buf_target_value.reverse()

                    # 用于训练的值
                    buf_state, buf_action = np.array(buf_state), np.array(buf_action)
                    buf_target_value = np.vstack(buf_target_value)

                    feed_dict = {
                        self.AC.state: buf_state,
                        self.AC.action: buf_action,
                        self.AC.target_value: buf_target_value,
                    }

                    buf_state, buf_action, buf_reward = [], [], []

                    # 做Sync
                    self.AC.update_global(feed_dict)
                    self.AC.pull_global()

                # 更新state和step
                state = new_state
                total_step += 1

                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


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

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
