#! /usr/bin/env python

import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.distributions import Categorical

# 策略网络的构造
class PolicyNet(nn.Module):

    def __init__(self, state_dim, nacts):
        super().__init__()

        # 特征提取层
        self.featnet = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 策略层
        self.pnet = nn.Linear(256, nacts)

    def forward(self, x):
        feat = self.featnet(x)
        logits = self.pnet(feat)
        return logits

    # 动作决策
    def act(self, x):
        with torch.no_grad():
            logits = self(x)
            dist = Categorical(logits=logits)
            return dist.sample().cpu().item()

class ActionBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.buffer.clear()

    def push(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def sample(self):
        # 轨迹采样
        state, action, reward, done = \
            zip(*self.buffer)
        reward = np.stack(reward, 0)
        # 计算回报函数
        for i in reversed(range(len(reward)-1)):
            reward[i] = reward[i] + GAMMA*reward[i+1]
        # 减去平均值，提高稳定性
        reward = reward - np.mean(reward)
        return np.stack(state, 0), np.stack(action, 0), reward, np.stack(done, 0)

    def __len__(self):
        return len(self.buffer)

def train(buffer, pnet, optimizer):
    # 获取训练数据
    state, action, reward, _ = buffer.sample()
    state = torch.tensor(state, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()

    # 计算损失函数
    logits = pnet(state)
    dist = Categorical(logits=logits)
    lossp = -(reward*dist.log_prob(action)).mean()
    
    optimizer.zero_grad()
    lossp.backward()
    optimizer.step()
    return lossp.item()

BATCH_SIZE = 16
NSTEPS = 1000000
GAMMA = 0.99
env = gym.make("CartPole-v0")
buffer = ActionBuffer(BATCH_SIZE)
pnet = PolicyNet(env.observation_space.shape[0], env.action_space.n)
pnet.cuda()
optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-3)

all_rewards = []
all_losses = []
episode_reward = 0
loss = 0.0

state = env.reset()
for nstep in range(NSTEPS):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = pnet.act(state_t)
    next_state, reward, done, _ = env.step(action)
    buffer.push(state, action, reward, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if done or len(buffer) == BATCH_SIZE:
        loss = train(buffer, pnet, optimizer)
        buffer.reset()