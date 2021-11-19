#! /usr/bin/env python

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from PIL import Image
import numpy as np
from collections import deque
from ex_5_8 import KFACOptimizer

class ActionBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.buffer.clear()

    def push(self, state, action, value, reward, done):
        self.buffer.append((state, action, value, reward, done))

    def sample(self, next_value):
        state, action, value, reward, done = \
            zip(*self.buffer)

        value = np.array(value + (next_value, ))
        done = np.array(done).astype(np.float32)
        reward = np.array(reward).astype(np.float32)
        delta = reward + GAMMA*(1-done)*value[1:] - value[:-1]

        rtn = np.zeros_like(delta).astype(np.float32)
        adv = np.zeros_like(delta).astype(np.float32)

        reward_t = next_value
        delta_t = 0.0

        for i in reversed(range(len(reward))):
            reward_t = reward[i] + GAMMA*(1.0 - done[i])*reward_t
            delta_t = delta[i] + (GAMMA*LAMBDA)*(1.0 - done[i])*delta_t
            rtn[i] = reward_t
            adv[i] = delta_t
        
        return np.stack(state, 0), np.stack(action, 0), rtn, adv

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
        img = img.crop((0, 30, 160, 200))
        img = img.resize((84, 84))
        img = np.array(img)/256.0
        return img - np.mean(img)

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

# 合并演员-评论家网络
class ActorCritic(nn.Module):

    def __init__(self, img_size, num_actions):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size
        self.num_actions = num_actions

        # 特征提取层
        # 对于Atari环境，输入为(4, 84, 84)
        self.featnet = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        gain = nn.init.calculate_gain('relu')
        # 演员网络第一层
        self.pnet1 = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
        )

        # 评论家网络第一层
        self.vnet1 = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU()
        )
        self._init(self.featnet, gain)
        self._init(self.pnet1, gain)
        self._init(self.vnet1, gain)
        
        gain = 1.0
        # 演员网络第二层
        self.pnet2 = nn.Linear(512, self.num_actions)
        # 评论家网络第二层
        self.vnet2 = nn.Linear(512, 1)
        self._init(self.vnet2, gain)
        self._init(self.pnet2, gain)

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def _init(self, mod, gain):
        for m in mod.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.featnet(x)
        logits = self.pnet2(self.pnet1(feat))
        value = self.vnet2(self.vnet1(feat))
        return logits, value

    def act(self, x):
        with torch.no_grad():
            logits, val = self(x)
            m = Categorical(logits=logits).sample().squeeze()
        return m.cpu().item(), val.cpu().item()

    def val(self, x):
        with torch.no_grad():
            _, val = self(x)
        return val.squeeze().cpu().item()

def train(buffer, next_value, pvnet, optimizer, use_gae=False):
    # 获取对应的采样数据
    state, action, rtn, adv = buffer.sample(next_value)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).cuda()
    rtn = torch.tensor(rtn, dtype=torch.float32).cuda()
    adv = torch.tensor(adv, dtype=torch.float32).cuda()

    # 计算未归一的概率和对应的价值
    logits, values = pvnet(state)

    if not use_gae:
        adv = (rtn - values).detach()

    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(action)
    lossp = -(adv*log_prob).mean() 
    lossv = 0.5*(rtn-values).pow(2).mean()
    
    optimizer.zero_grad()
    # 计算策略网络的费雪矩阵需要的梯度
    policy_fisher_loss = -log_prob.mean()
    with torch.no_grad():
      sampled_values = (values + torch.randn_like(values))
    # 计算价值网络的费雪矩阵需要的梯度
    value_fisher_loss = -(values - sampled_values).pow(2).mean()
    fisher_loss = policy_fisher_loss + value_fisher_loss

    # 计算费雪矩阵
    optimizer.acc_stats = True
    fisher_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(pvnet.parameters(), 0.5)
    optimizer.acc_stats = False
    optimizer.zero_grad()
    loss = lossp + lossv - REG*dist.entropy().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pvnet.parameters(), 0.5)
    optimizer.step()
    return lossp.item()

GAMMA = 0.99
LAMBDA = 0.95
NFRAMES = 4
BATCH_SIZE = 32
NSTEPS = 10000000
REG = 0.01
env = gym.make('PongDeterministic-v4')
env = EnvWrapper(env, NFRAMES)

state = env.reset()
buffer = ActionBuffer(BATCH_SIZE)
pvnet = ActorCritic((4, 84, 84), env.env.action_space.n)
pvnet.cuda()
optimizer = KFACOptimizer(pvnet, lr=1e-2)

all_rewards = []
all_losses = []
all_values = []
episode_reward = 0

loss = 0.0

for nstep in range(NSTEPS):

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action, value = pvnet.act(state_t)
    next_state, reward, done, _ = env.step(action)
    buffer.push(state, action, value, reward, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).cuda()
        all_values.append(pvnet.val(state_t))
        episode_reward = 0

    if done or len(buffer) == BATCH_SIZE:
        with torch.no_grad():
            state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).cuda()
            next_value = pvnet.val(state_t)

        loss = 0.9*loss + 0.1*train(buffer, next_value, pvnet, optimizer, True)
        buffer.reset()
