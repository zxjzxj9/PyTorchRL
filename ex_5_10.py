#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from collections import deque
import random
import numpy as np
import gym

class PolicyNet(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        # 特征提取网络
        self.featnet = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # 连续分布期望和对数标准差网络
        self.pnet_mu = nn.Linear(256, act_dim)
        self.pnet_logs = nn.Linear(256, act_dim)
    
    def forward(self, x):
        feat = self.featnet(x)
        mu = self.pnet_mu(feat)
        sigma = self.pnet_logs(feat).clamp(-20, 2).exp()
        # 根据期望和标准差得到正态分布
        return Independent(Normal(loc=mu, scale=sigma), reinterpreted_batch_ndims=1)
    
    def action_logp(self, x, reparam=False):
        dist = self(x)
        u = dist.rsample() if reparam else dist.sample()
        a = torch.tanh(u)
        # 概率密度变换
        # logp = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6).squeeze()
        logp = dist.log_prob(u) - (2*(np.log(2) - u - F.softplus(-2 * u))).squeeze()
        return a, logp

    def act(self, x):
        # 计算具体的动作
        with torch.no_grad():
            a, _ = self.action_logp(x)
            return a.cpu().item()

class DQN(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        # 特征提取网络
        self.featnet = nn.Sequential(
            nn.Linear(state_dim+act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # 价值网络
        self.vnet = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        feat = self.featnet(x)
        value = self.vnet(feat)
        return value

    def val(self, state, action):
        with torch.no_grad():
            value = self(state, action)
            return value.squeeze().cpu().item()

# 使用双网络方法来进行价值函数的预测
class TwinDQN(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.dqn1 = DQN(state_dim, act_dim)
        self.dqn2 = DQN(state_dim, act_dim)

    def update(self, other, polyak=0.995):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), other.parameters()):
                param1.data.copy_(polyak*param1.data+(1.0-polyak)*param2.data)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

class ExpReplayBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, bs):
        state, action, reward, next_state, done = \
            zip(*random.sample(self.buffer, bs))
        return np.stack(state, 0), np.stack(action, 0), \
            np.stack(reward, 0), np.stack(next_state, 0), \
            np.stack(done, 0).astype(np.float32)

    def __len__(self):
        return len(self.buffer)

def train(buffer, pnet, vnet, vnet_target, optim_p, optim_v):
    # 对经验进行采样
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda().unsqueeze(-1)
    action = torch.tensor(action, dtype=torch.float32).cuda().unsqueeze(-1)
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda().unsqueeze(-1)

    # 估算目标价值函数
    with torch.no_grad():
        next_action, logp = pnet.action_logp(next_state)
        next_qval1 = vnet_target.dqn1(next_state, next_action)
        next_qval2 = vnet_target.dqn2(next_state, next_action)
        next_qval = torch.min(next_qval1, next_qval2)
        target = reward + GAMMA * (1 - done) * (next_qval - REG * logp)

    # 计算价值网络损失函数
    value1 = vnet.dqn1(state, action)
    value2 = vnet.dqn2(state, action)
    lossv = 0.5*(value1 - target).pow(2).mean() + 0.5*(value2 - target).pow(2).mean()
    optim_v.zero_grad()
    lossv.backward()
    torch.nn.utils.clip_grad_value_(vnet.parameters(), 1.0)
    optim_v.step()

    # 计算策略网络损失函数，注意关闭价值网络参数梯度
    for param in vnet.parameters():
        param.requires_grad = False

    action, logp = pnet.action_logp(state, True)
    qval = torch.min(vnet.dqn1(state, action), vnet.dqn2(state, action))
    lossp = -torch.mean(qval - REG * logp)
    optim_p.zero_grad()
    lossp.backward()
    torch.nn.utils.clip_grad_value_(pnet.parameters(), 1.0)
    optim_p.step()

    for param in vnet.parameters():
        param.requires_grad = True

    vnet_target.update(vnet)
    return logp.mean().item()

BATCH_SIZE = 64
NSTEPS = 1000000
NBUFFER = 100000
GAMMA = 0.99
REG = 0.1
env = NormalizedActions(gym.make("Pendulum-v0"))
buffer = ExpReplayBuffer(NBUFFER)
pnet = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0])
vnet = TwinDQN(env.observation_space.shape[0], env.action_space.shape[0])
vnet_target = TwinDQN(env.observation_space.shape[0], env.action_space.shape[0])
pnet.cuda()
vnet.cuda()
vnet_target.cuda()
vnet_target.load_state_dict(vnet.state_dict())
optim_p = torch.optim.Adam(pnet.parameters(), lr=1e-3)
optim_v = torch.optim.Adam(vnet.parameters(), lr=1e-3)

all_rewards = []
all_losses = []
episode_reward = 0
loss = 0.0

state = env.reset()
for nstep in range(NSTEPS):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = pnet.act(state_t)
    next_state, reward, done, _ = env.step(action)
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(buffer) >= BATCH_SIZE:
        loss = train(buffer, pnet, vnet, vnet_target, optim_p, optim_v)