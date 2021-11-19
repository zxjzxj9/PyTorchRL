#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from PIL import Image

# DQN深度模型，用来估计Atari环境的Q函数
class DQN(nn.Module):

    def __init__(self, img_size, num_actions):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size
        self.num_actions = num_actions

        # 对于Atari环境，输入为(4, 84, 84)
        self.featnet = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 值网络，根据特征输出每个动作的价值
        self.vnet = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
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
        
        # 获取所有可能动作的价值
        values = self.vnet(feat)
        return values

    def act(self, x, epsilon=0.0):
        # ε-贪心算法
        if random.random() > epsilon:
            with torch.no_grad():
                values = self.forward(x)
            return values.argmax(-1).squeeze().item()
        else:
            return random.randint(0, self.num_actions-1)

from collections import deque
from heapq import heappush, heappushpop, heapify, nlargest
from operator import itemgetter

class Sample(tuple):
    def __lt__(self, x):
        return self[0] < x[0]

class PrioritizedExpReplayBuffer(object):

    def __init__(self, buffer_size, alpha):
        super().__init__()
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.buffer = []

    def heapify(self):
        heapify(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # 设置样本的初始时序误差
        td = 1.0 if not self.buffer else \
            nlargest(1, self.buffer, key=itemgetter(0))[0][0]

        # 向优先队列插入样本
        if len(self.buffer) < self.buffer_size:
            heappush(self.buffer, \
                Sample((td, state, action, reward, next_state, done)))
        else:
            heappushpop(self.buffer, \
                Sample((td, state, action, reward, next_state, done)))

    # 设置样本的时序误差
    def set_td_value(self, index, value):
        for idx_s, idx_t in enumerate(index):
            self.buffer[idx_t] = Sample((value[idx_s], *self.buffer[idx_t][1:]))

    def sample(self, bs, beta=1.0):
        # 计算权重并且归一化
        with torch.no_grad():
            weights = torch.tensor([val[0] for val in self.buffer])
            weights = weights.abs().pow(self.alpha)
            weights = weights/weights.sum()
            prob = weights.cpu().numpy()
            weights = (len(weights)*weights).pow(-beta)
            weights = weights/weights.max()
            weights = weights.cpu().numpy()
        index = random.choices(range(len(weights)), weights=prob, k=bs)

        # 根据index返回训练样本
        _, state, action, reward, next_state, done = \
            zip(*[self.buffer[i] for i in index])
        weights = [weights[i] for i in index]

        return np.stack(weights, 0).astype(np.float32), index, \
            np.stack(state, 0), np.stack(action, 0), \
            np.stack(reward, 0), np.stack(next_state, 0), \
            np.stack(done, 0).astype(np.float32)
            
    def __len__(self):
        return len(self.buffer)

class EnvWrapper(object):

    def __init__(self, env, num_frames):
        super().__init__()
        self.env_ = env
        self.num_frames = num_frames
        self.frame = deque(maxlen=num_frames)

    def _preprocess(self, img):
        # 预处理数据
        img = Image.fromarray(img)
        img = img.convert("L")
        img = img.resize((84, 84))
        return np.array(img)/256.0

    def reset(self):
        obs = self.env_.reset()
        for _ in range(self.num_frames):
            self.frame.append(self._preprocess(obs))
        return np.stack(self.frame, 0)

    def step(self, action):
        obs, reward, done, _ = self.env_.step(action)
        self.frame.append(self._preprocess(obs))
        return np.stack(self.frame, 0), np.sign(reward), done, {}
    
    @property
    def env(self):
        return self.env_

def train(buffer, model1, model2, optimizer):
    # 对经验回放的数据进行采样
    weights, index, state, action, reward, next_state, done = buffer.sample(BATCH_SIZE, BETA)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()
    weights = torch.tensor(weights, dtype=torch.float32).cuda()

    # 下一步状态的预测
    with torch.no_grad():
        # 用Q1计算最大价值的动作
        next_action = model1(next_state).argmax(-1)
        # 用Q2计算对应的最大价值
        target  = model2(next_state)\
            .gather(1, next_action.unsqueeze(-1)).squeeze()
        target = reward + (1-done)*GAMMA*target
    # 当前状态的预测

    predict = model1(state).gather(1, action.unsqueeze(-1)).squeeze()
    # 计算时序差分误差
    with torch.no_grad():
        td = (predict - target).squeeze().abs().cpu().numpy() + 1e-6
    buffer.set_td_value(index, td)

    loss = (weights*(predict - target).pow(2)).mean()
    # 损失函数的优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

GAMMA = 0.99
EPSILON_MIN = 0.01
EPSILON_MAX = 1.00
NFRAMES = 4
BATCH_SIZE = 32
NSTEPS = 4000000
NBUFFER = 20000
ALPHA = 0.4
BETA = 0.6
env = gym.make('PongDeterministic-v4')
env = EnvWrapper(env, NFRAMES)

state = env.reset()
buffer = PrioritizedExpReplayBuffer(NBUFFER, ALPHA)
# 构造两个相同的神经网络
dqn1 = DQN((4, 84, 84), env.env.action_space.n)
dqn2 = DQN((4, 84, 84), env.env.action_space.n)
dqn2.load_state_dict(dqn1.state_dict())
dqn1.cuda()
dqn2.cuda()
optimizer = torch.optim.Adam(dqn1.parameters(), 1e-4)

all_rewards = []
all_losses = []
episode_reward = 0

eps = lambda t: EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN)*np.exp(-t/30000)

for nstep in range(NSTEPS):
    p = eps(nstep)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = dqn1.act(state_t, p)
    next_state, reward, done, _ = env.step(action)

    buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(buffer) >= 10000:
        loss = train(buffer, dqn1, dqn2, optimizer)

    # 更新Q2参数
    if (nstep + 1) % 1000 == 0:
        dqn2.load_state_dict(dqn1.state_dict())

    # 重建二叉堆
    if (nstep + 1) % 100000 == 0:
        buffer.heapify()