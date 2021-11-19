#! /usr/bin/env python

from random import sample
from gym import Env, Space

class ExampleEnv(Env):

    # 强化学习环境初始化代码
    def __init__(self):
        super().__init__()

        # 初始化代码
        # self.action_space = ...
        # self.observation_space = ...
        self.reset()
    
    # 重置强化学习环境 
    def reset(self):
        # 重置的相关代码
        # obs = ...
        return obs

    # 智能体执行动作
    def step(self, action):
        # obs = ...
        # reward = ...
        # done = ...
        # info = ...
        return (obs, reward, done, info)

    # 渲染环境
    def render(self, mode="human"):
        # mode == "human" 代表图形界面窗口环境
        # mode == "rgb_array" 代表返回RGB图像
        # img = ...
        return img


class ExampleSpace(Space):

    # 初始化状态空间或者动作空间
    def __init__(self):
        super().__init__()

        # 相关初始化代码，比如初始化所有空间状态

    # 在状态或者动作空间中进行采样
    def sample(self):
        # sample = ...
        return sample

    # 检测某一状态或者动作是否在空间中
    def contains(self, x):
        # is_valid = ...
        return is_valid
    