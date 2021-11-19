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

        # 优势函数网络，根据特征输出每个动作的价值
        self.adv_net = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        # 价值函数网络，根据特征输出当前的状态的价值
        self.val_net = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
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
        values = self.val_net(feat) + self.adv_net(feat) - \
            self.adv_net(feat).mean(-1, keepdim=True)

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
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()
    # 下一步状态的预测，直接使用Q2的结果
    with torch.no_grad():
      target, _  = model2(next_state).max(-1)
      target = reward + (1-done)*GAMMA*target
    # 当前状态的预测
    predict = model1(state).gather(1, action.unsqueeze(-1)).squeeze()
    loss = (predict - target).pow(2).mean()
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
NBUFFER = 100000
env = gym.make('PongDeterministic-v4')
env = EnvWrapper(env, NFRAMES)

state = env.reset()
buffer = ExpReplayBuffer(NBUFFER)
dqn1 = DDQN((4, 84, 84), env.env.action_space.n)
dqn2 = DDQN((4, 84, 84), env.env.action_space.n)
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