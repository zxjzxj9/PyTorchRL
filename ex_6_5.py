#! /usr/bin/env python

import math
import random
import numpy as np
from scipy.special import softmax
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_gomoku.envs.gomoku import  GomokuState, Board
from gym_gomoku.envs.util import gomoku_util

class PolicyValueNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()

        self.featnet = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
        )

        self.pnet = nn.Sequential(
            nn.Conv2d(128, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*board_size*board_size, board_size*board_size)
        )

        self.vnet = nn.Sequential(
            nn.Conv2d(128, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*board_size*board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.featnet(x)
        prob = self.pnet(feat).softmax(-1)
        val = self.vnet(feat).tanh()
        return prob, val

    def evaluate(self, x):
        with torch.no_grad():
            prob, val = self(x)
            return prob.squeeze(), val.squeeze()

class TreeNode(object):
    def __init__(self, parent, prior):
        self.parent = parent
        self.prior = prior

        self.Q = 0
        self.N = 0
        self.children = {}

    def score(self, c_puct):
        sqrt_sum = np.sqrt(np.sum([node.N for node in self.parent.children.values()]))
        return self.Q + c_puct*self.prior*sqrt_sum/(1 + self.N)

    def update(self, qval):
        self.Q = self.Q*self.N + qval
        self.N += 1
        self.Q = self.Q/self.N

    def backup(self, qval):
        self.update(qval)
        if self.parent: 
            self.parent.backup(-qval)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda x: x[1].score(c_puct))

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = TreeNode(self, prior)

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_leaf(self):
        return len(self.children) == 0

class MCTSBot(object):
    def __init__(self, board_size, c_puct=5.0, nsearch=2000):

        self.board_size = board_size
        self.root = TreeNode(None, 1.0)
        self.c_puct = c_puct
        self.nsearch = nsearch

    def get_feature(self, board, player):
        feat = board.encode()
        feat1 = (feat == 1).astype(np.float32)
        feat2 = (feat == 2).astype(np.float32)
        feat3 = np.zeros((self.board_size, self.board_size)).astype(np.float32)
        if board.last_action is not None:
            x, y = board.action_to_coord(board.last_action)
            feat3[x, y] = 1.0
        if player == 'white':
            feat4 = np.zeros((self.board_size, self.board_size)).astype(np.float32)
            return np.stack([feat1, feat2, feat3, feat4], axis=0)
        elif player == 'black':
            feat4 = np.ones((self.board_size, self.board_size)).astype(np.float32)
            return np.stack([feat1, feat2, feat3, feat4], axis=0)

    def mcts_search(self, state, pvnet):
        node = self.root

        while not node.is_leaf:
            action, node = node.select(self.c_puct)
            state = state.act(action)

        feature = self.get_feature(state.board, state.color)
        feature = torch.tensor(feature).unsqueeze(0)
        probs, val = pvnet.evaluate(feature)
        actions = state.board.get_legal_action()
        probs = probs[actions]

        if state.board.is_terminal():
            _, win_color = \
                gomoku_util.check_five_in_row(state.board.board_state)
            if win_color == 'empty':
                val = 0.0
            elif win_color == state.color:
                val = 1.0
            else:
                val = -1.0
        else:
            node.expand(actions, probs)
        
        node.backup(-val)

    def alpha(self, state, pvnet, temperature=1e-3):

        for _ in range(self.nsearch):
            self.mcts_search(state, pvnet)

        node_info = [(action, node.N) for action, node in self.root.children.items()]
        actions, nvisits = zip(*node_info)
        actions = np.array(actions)
        probs = np.log(np.array(nvisits)+1e-6)/temperature
        probs = softmax(probs)
        return actions, probs

    def reset(self):
        self.root = TreeNode(None, 1.0)

    def step(self, action):
        self.root = self.root.children[action]
        self.root.parent= None

class MCTSRunner(object):
    def __init__(self, board_size, pvnet, 
        eps = 0.25, alpha = 0.03, 
        c_puct=5.0, nsearch=2000, selfplay=False):
        self.pvnet = pvnet
        self.mctsbot = MCTSBot(board_size, c_puct, nsearch)
        self.board_size = board_size
        self.selfplay = selfplay
        self.eps = eps
        self.alpha = alpha

    def reset(self):
        self.mctsbot.reset()

    def play(self, state, temperature=1e-3, return_data=False):

        probs = np.zeros(self.board_size*self.board_size)
        feat = self.mctsbot.get_feature(state.board, state.color)

        a, p = self.mctsbot.alpha(state, self.pvnet, temperature)
        probs[a] = p

        action = -1
        if self.selfplay:
            p = (1 - self.eps)*p + self.eps*np.random.dirichlet([self.alpha]*len(a))
            action = np.random.choice(a, p=p)
            self.mctsbot.step(action)
        else:
            action = np.random.choice(a, p=p)
            self.mctsbot.reset() 

        if return_data:
            return action, feat, probs
        else:
            return action

class MCTSTrainer(object):
    def __init__(self):
        self.board_size = 9
        self.buffer_size = 10000
        self.c_puct = 5.0
        self.nsearch = 1000
        self.temperature = 1e-3

        self.lr = 1e-3
        self.l2_reg = 1e-4
        self.niter = 5
        self.batch_size = 128
        self.ntrain = 1000

        self.buffer = deque(maxlen=self.buffer_size)
        self.pvnet = PolicyValueNet(self.board_size)
        self.optimizer = torch.optim.Adam(self.pvnet.parameters(), 
            lr=self.lr, weight_decay=self.l2_reg)
        self.mcts_runner = MCTSRunner(self.board_size, self.pvnet, 
                c_puct=self.c_puct, nsearch=self.nsearch, selfplay=True)

    def reset_state(self):
        self.state = GomokuState(Board(self.board_size), gomoku_util.BLACK)

    def collect_data(self):
        self.reset_state()
        self.mcts_runner.reset()

        feats = []
        probs = []
        players = []
        values = []
        cnt = 0
        while True:
            print(f"step {cnt+1}"); cnt += 1
            # print(self.state)
            action, feat, prob = self.mcts_runner.play(self.state, self.temperature, True)
            feats.append(feat)
            probs.append(prob)
            players.append(self.state.color)
            self.state = self.state.act(action)

            if self.state.board.is_terminal():
                _, win_color = \
                    gomoku_util.check_five_in_row(self.state.board.board_state)
                if win_color == 'empty':
                    values = [0.0]*len(players)
                else:
                    values = [1.0 if player == win_color else -1.0 for player in players]

                return zip(feats, probs, values)

    def data_augment(self, data):
        ret = []
        for feat, prob, value in data:
            for i in range(0, 4):
                feat = np.rot90(feat, i, (1, 2))
                ret.append((feat, prob, value))
                ret.append((feat[:,::-1,:], prob, value))
                ret.append((feat[:,:,::-1], prob, value))
        return ret

    def train_step(self):
        data = self.collect_data()
        data = self.data_augment(data)
        self.buffer.extend(data)

        for idx in range(self.niter):
            feats, probs, values = zip(*random.sample(self.buffer, self.batch_size))
            feats = torch.tensor(np.stack(feats, axis=0))
            probs = torch.tensor(np.stack(probs, axis=0))
            values = torch.tensor(np.stack(values, axis=0))

            p, v = self.pvnet(feats)
            loss = (v - values).pow(2).mean() - (probs*(p + 1e-6).log()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            print(f"In iteration: {idx}, Loss function: {loss.item():12.6f}")
            self.optimizer.step()

    def train(self):
        for idx in range(self.ntrain):
            print(f"In training step {idx}")
            self.train_step()

if __name__ == "__main__":
    trainer = MCTSTrainer()
    trainer.train()