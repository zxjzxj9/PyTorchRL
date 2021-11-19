#! /usr/bin/env python

from gym import Env, spaces
import random
import math

# 3x3的井字棋棋盘
NX = 3
NY = 3

class TicTacToe(Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(NX*NY)
        self.observation_space = spaces.Discrete(NX*NY)
        self.symbols = {
            0: '*',
            1: 'x',
            2: '+'
        }
        self.reset()

    def reset(self):
        self.board = [[0 for _ in range(NX)] for _ in range(NY)]
        self.reward = 0
        self.nstep = 0
        self.first_player = True
        self.end = False
        return self.board

    def sample(self):
        ret = []
        for action in range(NX*NY):
            x = action%3
            y = action/3
            if self.board[x][y] == 0: ret.append(action)
        return random.choice(ret)

    def step(self, action):

        if self.end:
            return self.board, 0, self.end, {}

        x = action%3
        y = action//3

        # 无效的操作，设置负无穷的奖励
        if self.board[x][y] != 0:
            self.end = True
            return self.board, -math.inf, self.end, {}

        if self.first_player:
            self.board[x][y] = 1
        else:
            self.board[x][y] = 2

        self.first_player = not self.first_player
        self.nstep += 1

        # 验证游戏结束
        if self.nstep >= 5:
            lines = [line for line in self.lines if (x, y) in line]
            for line in lines:
                x1, y1 = line[0][0], line[0][1]
                x2, y2 = line[1][0], line[1][1]
                x3, y3 = line[2][0], line[2][1]
                if self.board[x1][y1] == self.board[x2][y2] and \
                    self.board[x1][y1] == self.board[x3][y3]:
                    self.end = True
                    break
            if self.end and self.board[x][y] == 1:
                self.reward = +1
                return self.board, self.reward, self.end, {}
            if self.end and self.board[x][y] == 2:
                self.reward = -1
                return self.board, self.reward, self.end, {}

        # 平局的情况
        if self.nstep >= 9:
            self.reward = 0
            self.end = True
            return self.board, self.reward, self.end, {}

        return self.board, 0, self.end, {}

    def render(self, mode=None):
        ret = ""
        for i in range(3):
            ret += " | ".join([self.symbols[j] for j in self.board[i]])
            ret += "\n"
            if i != 2: ret += "---"*3
            ret += "\n"
        return ret
