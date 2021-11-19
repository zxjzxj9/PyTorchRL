#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from PIL import Image

class CDQN(nn.Module):
    def __init__(self, img_size, num_actions, vmin, vmax, num_cats):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size
        self.num_actions = num_actions
        self.num_cats = num_cats
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

        self.category_net = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions*self.num_cats),
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
        
        # 获取所有可能动作的价值概率分布
        logits = self.category_net(feat)\
            .view(-1, self.num_actions, self.num_cats)

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

def train(buffer, model, optimizer):
    # 对经验回放的数据进行采样
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()
    idx = torch.arange(BATCH_SIZE).cuda()

    # 下一步状态的预测
    with torch.no_grad():
        prob = model(next_state).softmax(-1)
        value_dist = prob*model.vrange
        next_action = value_dist.sum(-1).argmax(-1)
        prob = prob[idx, next_action[idx], :]
        # 计算下一步奖励的映射
        value = reward.unsqueeze(-1) + \
          (1-done).unsqueeze(-1)*GAMMA*model.vrange.squeeze(0)
        value = (value.clamp(VMIN, VMAX) - VMIN)/DV
        lf, uf = value.floor(), value.ceil()
        ll, ul = lf.long(), uf.long()
        target = torch.zeros_like(value)

        target.scatter_add_(1, ll, prob*(uf-value))
        target.scatter_add_(1, ul, prob*(value-lf))

    # 当前状态的预测
    predict = model(state)[idx, action[idx], :]
    loss = -(target*predict.log_softmax(-1)).mean()
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
VMIN = -10
VMAX = 10
NCATS = 51
DV = (VMAX - VMIN)/(NCATS - 1)
env = gym.make('PongDeterministic-v4')
env = EnvWrapper(env, NFRAMES)

# print(env.reset().shape)
# print(env.step(1)[0].shape)

state = env.reset()
buffer = ExpReplayBuffer(NBUFFER)
dqn = CDQN((4, 84, 84), env.env.action_space.n, VMIN, VMAX, NCATS)
dqn.cuda()
optimizer = torch.optim.Adam(dqn.parameters(), 1e-4)

all_rewards = []
all_losses = []
episode_reward = 0

eps = lambda t: EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN)*np.exp(-t/30000)

for nstep in range(NSTEPS):
    p = eps(nstep)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = dqn.act(state_t, p)
    next_state, reward, done, _ = env.step(action)
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(buffer) >= 10000:
        loss = train(buffer, dqn, optimizer)