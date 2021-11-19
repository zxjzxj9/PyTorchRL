#! /usr/bin/env python
from math import radians
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class Env(object):
    def __init__(self, bandit):
        # 定义摇臂数目
        self.nbandit = len(bandit)
        self.bandit = torch.tensor(bandit)
        
    def sample(self, idx):
        rewards = self.bandit + torch.randn_like(self.bandit)
        return rewards[idx]

class EGreedyAgent(object):
    def __init__(self, env, epislon=0.0, gamma=0.9):
        self.env = env
        self.value = torch.zeros(env.nbandit)
        # 定义探索的概率， 其中概率0.0为贪心策略
        self.epislon = epislon
        # 定义折扣系数
        self.gamma = gamma

        self.cur = None
        self.rwd = None

    def sample(self):
        if random.random() < self.epislon:
            reward, idx = self.random_search()
        else:
            reward, idx = self.greedy_search()
            
        if self.cur is not None:
            self.value[self.cur] = self.rwd + self.gamma*self.value[idx]

        self.cur = idx
        self.rwd = reward
        return reward

    def greedy_search(self):
        with torch.no_grad():
            max_val = torch.max(self.value).item()
            max_idx = (self.value == max_val).nonzero(as_tuple=True)[0].tolist()
            idx = random.choice(max_idx)
            reward = env.sample(idx)
        return reward, idx

    def random_search(self):
        idx = random.choice(range(self.env.nbandit))
        reward = env.sample(idx)
        return reward, idx 

if __name__ == "__main__":
    env = Env([float(i) for i in range(1, 11)])
    def calc_rewards(env, epislon, nagents=1000, niter=100):
        agents = [EGreedyAgent(env, epislon) for _ in range(nagents)]
        ret = []
        for _ in range(niter):
            ret.append(float(np.mean([agent.sample() for agent in agents])))
        return ret
    plt.plot(calc_rewards(env, 0.0), ls="-", label="$\epsilon=0.00$")
    plt.plot(calc_rewards(env, 0.01), ls="--", label="$\epsilon=0.01$")
    plt.plot(calc_rewards(env, 0.1), ls=":", label="$\epsilon=0.10$")
    plt.legend()
    plt.savefig("ex_2_2.png")