#! /usr/bin/env python

import gym
import matplotlib.pyplot as plt

# 演示如何使用MuJoCo强化学习环
env = gym.make('Ant-v2')
obs = env.reset()
print(obs.shape)

# 观测空间的大小为111，动作空间的大小为8
print(env.observation_space)
print(env.action_space)

# 执行20个片段的采样，每次重置
for episode_idx in range(10):
    obs = env.reset()
    # 单次采样的路径，最多100步
    for _ in range(100):
        # 随机在动作空间进行采样
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(action, obs.shape, reward)
        # 如果强化学习环境处于结束状态，则跳出循环
        if done: break

img = env.render(mode='rgb_array')
plt.imshow(img)
plt.savefig("fig.3.7.png")