#! /usr/bin/env python

import gym
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class PolicyNet(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.featnet = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.pnet = nn.Linear(256, act_dim)
    
    def forward(self, x):
        feat = self.featnet(x)
        return self.pnet(feat).tanh()

    def act(self, x):
        with torch.no_grad():
            action = self(x)
            action += 0.1*torch.randn_like(action)
            return action.clamp(-1, 1).cpu().item()

    def update(self, other, polyak=0.995):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), other.parameters()):
                param1.data.copy_(polyak*param1.data+(1-polyak)*param2.data)

class DQN(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.featnet = nn.Sequential(
            nn.Linear(state_dim+act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

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
        
    def update(self, other, polyak=0.995):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), other.parameters()):
                param1.data.copy_(polyak*param1.data+(1-polyak)*param2.data)

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

def train(buffer, pnet, pnet_target, vnet, vnet_target, optim_p, optim_v):
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda().unsqueeze(-1)
    action = torch.tensor(action, dtype=torch.float32).cuda().unsqueeze(-1)
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda().unsqueeze(-1)

    with torch.no_grad():
        next_action = pnet_target(next_state)
        next_qval = vnet_target(next_state, next_action)
        target = reward + GAMMA * (1 - done) * next_qval

    value = vnet(state, action)
    lossv = 0.5*(value - target).pow(2).mean()
    optim_v.zero_grad()
    lossv.backward()
    torch.nn.utils.clip_grad_value_(vnet.parameters(), 1.0)
    optim_v.step()

    for param in vnet.parameters():
        param.requires_grad = False

    qval = vnet(state, pnet(state))
    lossp = -torch.mean(qval)
    optim_p.zero_grad()
    lossp.backward()
    torch.nn.utils.clip_grad_value_(pnet.parameters(), 1.0)
    optim_p.step()

    for param in vnet.parameters():
        param.requires_grad = True

    vnet_target.update(vnet)
    pnet_target.update(pnet)
    return lossp

BATCH_SIZE = 64
NSTEPS = 1000000
NBUFFER = 100000
GAMMA = 0.99
REG = 0.1
env = NormalizedActions(gym.make("Pendulum-v0"))
buffer = ExpReplayBuffer(NBUFFER)

pnet = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0])
pnet_target = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0])
vnet = DQN(env.observation_space.shape[0], env.action_space.shape[0])
vnet_target = DQN(env.observation_space.shape[0], env.action_space.shape[0])
pnet.cuda()
pnet_target.cuda()
pnet_target.load_state_dict(pnet.state_dict())
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
        loss = train(buffer, pnet, pnet_target, vnet, vnet_target, optim_p, optim_v)
