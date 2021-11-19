#! /usr/bin/env python

import gym
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')

# 重置环境，返回当前的环境状态
obs = env.reset()
print(obs)

# 执行20个片段的采样，每次重置
for episode_idx in range(10):
    obs = env.reset()
    # 单次采样的路径，最多100步
    for _ in range(100):
        # 随机在动作空间进行采样
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(action, obs, reward)
        # 如果强化学习环境处于结束状态，则跳出循环
        if done: break

# 渲染相关代码，img为输出图像，img可以作为深度学习模型输入使用
img = env.render(mode='rgb_array')
plt.imshow(img)
plt.savefig("fig.3.6.png")