# Rein48 (Draft 未完成)

2048 is a very simply and popular game, which can be a perfect example of learning Deep Reinforcement Learning. In this project, We train computer to master the game of 2048 using Deep Reinforcement Learning without human knowledge.

2048是一款非常简单而且流行的手机游戏，非常适合作为深度增强学习的训练环境。在这个项目中，我计划训练电脑自动学会玩2048.

There are some works about reinforcement learning have been done for 2048, using the method of DQN or another evolutionary approaching. We apply our A3C approach, which is the state-of-the-art method until now, to master this game.

已经有很多针对2048的强化学习实现，这些方法基于DQN或者遗传算法等。这里我们用最先进的A3C算法完成2048的强化学习。

# Introduction 介绍

First, we finish the simplest implement of the game 2048 with Python 3, which means we only release the terminal version.

首先，我们要完成了简单的2048实现，这里并没有图形化界面，图像显示在命令行中。

2048的优势在于时效性很强的Reward，同时有很简单的Action Space，但是State Space比较大。

初始状态时，移动方格的Policy是Norm Random的（上下左右各25%）。用GPU跑几万个例子出来做Data。
鉴于2048的方格结构，可以考虑使用CNN或者ResNet。

可以用ResNet做出Grid的特征抽取，然后结合ReLU和Batch Normalization来训练NN。
# Install 安装

# Detail 实现方法

## 1.

# Resource 相关资料

1.The paper of A3C : 
> Asynchronous Methods for Deep Reinforcement Learning
