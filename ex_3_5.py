#! /usr/bin/env python

import gym
import matplotlib.pyplot as plt

# 演示如何使用Atari强化学习环境

env = gym.make('SpaceInvaders-v0')
obs = env.reset()

# 观测空间的大小为210x160x3 
print(obs.shape)
print(env.action_space)

# 执行20个片段的采样，每次重置
for episode_idx in range(10):
    obs = env.reset()
    # 单次采样的路径，最多100步
    for _ in range(100):
        # 随机在动作空间进行采样
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(action, obs.shape, reward)
        # 如果强化学习环境处于结束状态，则跳出循环
        if done: break


# 渲染相关代码，img为输出图像，img可以作为深度学习模型输入使用
plt.imshow(obs)
plt.savefig("fig.3.6.png")