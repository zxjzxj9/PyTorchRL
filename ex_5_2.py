#! /usr/bin/env python

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch.distributions import Normal

class PolicyNet(nn.Module):

    def __init__(self, state_dim, act_dim, act_min, act_max):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.featnet = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.pnet_mu = nn.Linear(256, act_dim)
        self.pnet_logs = nn.Parameter(torch.zeros(act_dim))

        self.act_min = act_min
        self.act_max = act_max
    
    def forward(self, x):
        feat = self.featnet(x)
        mu = (self.act_max - self.act_min)*\
            self.pnet_mu(feat).sigmoid() + self.act_min
        sigma = (F.softplus(self.pnet_logs) + 1e-4).sqrt()
        return mu, sigma

    def act(self, x):
        with torch.no_grad():
            mu, sigma = self(x)
            dist = Normal(mu, sigma)
            return dist.sample()\
                .clamp(self.act_min, self.act_max)\
                .squeeze().cpu().item()

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
    action = torch.tensor(action, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()

    # 计算损失函数
    mu, sigma = pnet(state)
    dist = Normal(mu, sigma)
    lossp = -(reward*dist.log_prob(action)).mean()
    
    optimizer.zero_grad()
    lossp.backward()
    torch.nn.utils.clip_grad_norm_(pnet.parameters(), 50.0)
    optimizer.step()
    return lossp.item()

BATCH_SIZE = 16
NSTEPS = 1000000
GAMMA = 0.9
env = gym.make("Pendulum-v0")
buffer = ActionBuffer(BATCH_SIZE)
pnet = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0], 
                 env.action_space.low[0], env.action_space.high[0])
pnet.cuda()
optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-4)

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