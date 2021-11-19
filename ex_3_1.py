#! /usr/bin/env python
import random
import math
import copy
from functools import lru_cache

class TicTacToe(object):
    def __init__(self):
        super().__init__()
        # 所有可能连线
        self.lines = \
            [[(0, i), (1, i), (2, i)] for i in range(3)] + \
            [[(i, 0), (i, 1), (i, 2)] for i in range(3)] + \
            [[(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]    
        self.reset()

    def reset(self):
        self.board = [['*' for _ in range(3)] for _ in range(3)]
        self.nstep = 0
        self.first_player = True
        self.end = False
        self.reward = None

    def action_space(self):
        # 返回可能的动作空间
        ret = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '*': ret.append((i, j))
        return ret

    def step(self, pos):
        if self.end: return
        # 执行一步落子
        x, y = pos
        if self.first_player:
            self.board[x][y] = 'x'
        else:
            self.board[x][y] = 'o'

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
            if self.end and self.board[x][y] == 'x':
                self.reward = +1
                return self
            if self.end and self.board[x][y] == 'o':
                self.reward = -1
                return self

        # 平局的情况
        if self.nstep >= 9:
            self.reward = 0
            self.end = True
            return self

        return self

    def __str__(self):
        ret = ""
        for i in range(3):
            ret += " | ".join(self.board[i])
            ret += "\n"
            if i != 2: ret += "---"*3
            ret += "\n"
        return ret

    def __hash__(self):
        return self.__str__().__hash__()

# 执行极小化极大搜索算法
@lru_cache(100000)
def minimax_search(env):
    # 如果处于结束状态，直接返回目前的奖励
    if env.end: return env.reward, None
    # 第一个决策的智能体，极大化奖励
    if env.first_player:
        max_reward = -math.inf
        max_step = None
        for pos in env.action_space():
            reward, _ = minimax_search(copy.deepcopy(env).step(pos))
            if reward > max_reward:
                max_reward = reward
                max_step = pos
        return max_reward, max_step
    # 第二个决策的智能体，极小化奖励
    else:
        min_reward = +math.inf
        min_step = None
        for pos in env.action_space():
            reward, _ = minimax_search(copy.deepcopy(env).step(pos))
            if reward < min_reward:
                min_reward = reward
                min_step = pos
        return min_reward, min_step

if __name__ == "__main__":
    # 随机策略
    env = TicTacToe()
    print(env)
    while not env.end:
        env.step(random.choice(env.action_space()))
        print(env)
    print(env.reward)

    # 重置，然后用极小化极大搜索算法进行最优步骤搜索
    env.reset()
    env.step(random.choice(env.action_space()))
    env.step(random.choice(env.action_space()))
    print(env)
    reward, step = minimax_search(env)
    print(reward, step)