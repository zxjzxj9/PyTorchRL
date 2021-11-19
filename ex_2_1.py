#! /usr/bin/env python
import torch

class Env:
    def __init__(self, grid):
        self.grid = grid

    def iterate(self, gamma=0.9):
        # 存储上一步迭代的价值函数
        tmp = self.grid.clone()
        nx, ny = self.grid.shape
        for i in range(nx):
            for j in range(ny):
                vals = []
                # 右下角时只有一种接下来的状态(0, 0)
                if i == 3 and j == 3: vals.append(tmp[0, 0])
                else:
                    if i > 0: vals.append(tmp[i-1, j])
                    if i < 3: vals.append(tmp[i+1, j])
                    if j > 0: vals.append(tmp[i, j-1])
                    if j < 3: vals.append(tmp[i, j+1])
                # 获得所有可能状态的最大价值函数
                maxval = max(vals)
                if i == 3 and j == 3: self.grid[i, j] = 5.0 + gamma*maxval
                else: self.grid[i, j] = gamma*maxval
        err = (self.grid - tmp).abs().max().item()
        return self.grid, err

grid = torch.zeros(4, 4)
env = Env(grid)
for step in range(1000):
    _, err = env.iterate()
    if err < 1e-3:
        print(step, err)
        break
print(grid)
