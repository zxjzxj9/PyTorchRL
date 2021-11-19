#! /usr/bin/env python

import torch
import math


# 重要性采样模拟

# 均匀分布概率密度函数
def p(x):
    return ((x>=0)&(x<=1)).float()

# 标准正态分布概率密度函数
def q(y):
    return 1.0/math.sqrt(2*math.pi)*(-y.pow(2)/2).exp()

# 10万次采样
x = torch.rand(100000)
y = torch.randn(100000)

print("E(f(x))={}".format(x.pow(2).mean()))
print("E(f(y)*p(y)/q(y))={}"\
    .format((y.pow(2)*p(y)/q(y)).mean()))