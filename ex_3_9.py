#! /usr/bin/env python

import deepmind_lab
import numpy as np

num_episodes = 10 
# 关卡目录: lab/game_scripts/levels/
env = deepmind_lab.Lab('seekavoid_arena_01', ['RGB_INTERLEAVED'], 
    {'fps': '60', 'width': '640', 'height': '480'})
env.reset()

# 获取观测空间的信息
observation_spec = env.observation_spec()
print(observation_spec)

# 获取动作空间的信息
action_spec = env.action_spec()
print(action_spec)

# 获取动作空间的各种图像
obs = env.observations() 
rgb_i = obs['RGB_INTERLEAVED']
print(rgb_i)

# 选定向前运动的动作
action = np.zeros([7], dtype=np.intc)
action[3] = 1

score = 0
for _ in range(num_episodes):
    while env.is_running():
      # 连续执行4帧
      reward = env.step(action, num_steps=4)
      if reward != 0:
        score += reward
        print('Score =', score)