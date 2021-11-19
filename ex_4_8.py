#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from PIL import Image

# Duel DQN深度模型，用来估计Atari环境的Q函数
class DDQN(nn.Module):

    def __init__(self, img_size, num_actions, vmin, vmax, num_cats):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size
        self.num_actions = num_actions
        self.num_cats = num_cats

        # 最小的价值和最大的价值
        self.vmax = vmax
        self.vmin = vmin

        # 计算从vmin到vmax之间的一系列离散的价值
        self.register_buffer(
            "vrange",
            torch.linspace(self.vmin, self.vmax, num_cats)\
                .view(1, 1, -1)
        )

        # 计算两个价值的差值
        self.register_buffer(
            "dv",
            torch.tensor((vmax-vmin)/(num_cats-1))
        )

        # 对于Atari环境，输入为(4, 84, 84)
        self.featnet = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 优势函数网络，根据特征输出每个动作的价值
        self.adv_net = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions*self.num_cats)
        )

        # 价值函数网络，根据特征输出当前的状态的价值
        self.val_net = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_cats)
        )

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def forward(self, x):        
        bs = x.size(0)

        # 提取特征
        feat = self.featnet(x).view(bs, -1)
        
        val = self.val_net(feat).view(-1, 1, self.num_cats)
        adv = self.adv_net(feat).view(-1, self.num_actions, self.num_cats)
        logits = val + adv - adv.mean(1, keepdim=True)
        return logits

    def qval(self, x):
        probs = self.forward(x).softmax(-1)
        return (probs*self.vrange).sum(-1)

    def act(self, x, epsilon=0.0):
        # ε-贪心算法
        if random.random() > epsilon:
            with torch.no_grad():
                qval = self.qval(x)
            return qval.argmax(-1).squeeze().item()
        else:
            return random.randint(0, self.num_actions-1)
