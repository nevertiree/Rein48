# -*- coding: utf-8 -*-

import multiprocessing  # 用于统计CPU核数
import threading
import os
import shutil

from game.game_cli import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GAME = 'CartPole-v0'  # 游戏名称
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


class ACNet(object):
    def __init__(self, scope, global_ac=None):

        if scope == GLOBAL_NET_SCOPE:   # 全局网络
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'S')
                self.actor_params, self.critic_params = self._build_net(scope)[-2:]
        else:   # 本地网络，用于计算loss
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S, N_S], 'S')
                self.action = tf.placeholder(tf.int32, [None, ], 'A')
                self.target_value = tf.placeholder(tf.float32, [None, 1], 'Target_Value')

                self.action_prob, self.estimated_value, self.actor_params, self.critic_params = self._build_net(scope)

                # 根据TD Error计算Actor和Critic的Loss
                # TD Error = Target Value - Estimated Value
                # todo
                td_error = tf.subtract(self.target_value, self.estimated_value, name='TD_error')

                # 计算 critic loss
                with tf.name_scope('c_loss'):
                    # TD Error的平方和平均
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                # 计算 actor loss
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

                # 用两个 loss 计算要推送的 gradients
                with tf.name_scope('local_grad'):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

            with tf.name_scope('sync'):  # 同步
                # 从远程获取参数
                with tf.name_scope('pull'):
                    self.pull_actor_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.actor_params, global_ac.actor_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.critic_params, global_ac.critic_params)]
                # 向远程提交数据
                # todo tf.name_scope(push)
                with tf.name_scope('push'):
                    self.update_actor_op = OPT_A.apply_gradients(zip(self.actor_grads, global_ac.actor_params))
                    self.update_critic_op = OPT_C.apply_gradients(zip(self.critic_grads, global_ac.critic_params))

    def _build_net(self, scope):
        # 在这里搭建 Actor 和 Critic 的网络
        w_init = tf.random_normal_initializer(0., .1)

        # Actor网络 两层全连接
        with tf.variable_scope('actor'):
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

        # Critic网络 两层全连接
        with tf.variable_scope('critic'):
            flat_state = tf.reshape(self.state, [-1, 4 * 4 * 1])
            l_c = tf.layers.dense(flat_state, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            estimated_value = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='estimated_value')  # state value

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return action_prob, estimated_value, actor_params, critic_params

    def update_global(self, feed_dict):
        SESS.run([self.update_actor_op, self.update_critic_op], feed_dict=feed_dict)  # local grads applies to global net

    def pull_global(self):
        SESS.run([self.pull_actor_params_op, self.pull_c_params_op])

    # 根据state选择一个action
    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.action_prob, feed_dict={self.state: s[np.newaxis, :]})
        # ravel()行序优先拉平
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, global_ac):
        self.env = Game()
        self.name = name  # 自己的名字
        # 所服务的全局Actor-Critic
        self.AC = ACNet(name, global_ac)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        # 缓存, 用于 n_steps 更新
        buf_state, buf_action, buf_reward = [], [], []
        # GLOBAL_EP是迭代次数
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # 新开一个episode, reward清零
            state = np.array(self.env.reset())
            ep_r = 0
            # 迭代到一个episode结束
            while True:
                # 根据state做出一个action
                action = self.AC.choose_action(state)
                # 根据action得到下一步
                # new_state, reward, done, info = self.env.step(action)
                new_state, reward, done = self.env.step(action)
                new_state = np.array(new_state)
                # Done说明本episode结束
                if done:
                    reward = -5

                # 更新总的reward值
                ep_r += reward

                # 把state action reward 做一次记录，每个Iter清理一次
                # todo 改变了state的状态
                buf_state.append(state)
                buf_action.append(action)
                buf_reward.append(reward)

                # 每 UPDATE_GLOBAL_ITER 步 或者回合完了, 进行 sync 操作
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    # 获得用于计算 TD error的下一state的value
                    if done:
                        new_state_value = 0   # terminal
                        SCORE.append(np.sum(new_state))
                    else:
                        new_state_value = SESS.run(self.AC.estimated_value,
                                                   {self.AC.state: new_state[np.newaxis, :]})[0, 0]

                    # 进行 n_steps forward view
                    buf_target_value = []
                    for reward in buf_reward[::-1]:    # reverse buffer r
                        new_state_value = reward + GAMMA * new_state_value
                        buf_target_value.append(new_state_value)
                    buf_target_value.reverse()

                    buf_state, buf_action = np.array(buf_state), np.array(buf_action)
                    buf_target_value = np.vstack(buf_target_value)

                    feed_dict = {
                        self.AC.state: buf_state,
                        self.AC.action: buf_action,
                        self.AC.target_value: buf_target_value,
                    }
                    # 把本地的数据update到远程的AC
                    # todo update_global
                    self.AC.update_global(feed_dict)
                    # 清理每个Iter的buffer
                    buf_state, buf_action, buf_reward = [], [], []
                    # 从远程取得AC的新数据
                    self.AC.pull_global()

                # 更新s 和 step
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
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
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

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()

    plt.plot(np.arange(len(SCORE)), SCORE)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
#