#! /usr/bin/env python

import torch

mu = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(2.0, requires_grad=True)

dist = torch.distributions.Normal(mu, sigma)

nsample = 10000

xs = dist.rsample((nsample, )).detach()
xs_val = xs.pow(2)

# 直接求解方法
# loss = xs.pow(2).mean()
# 蒙特卡洛梯度估计
loss = (xs_val*dist.log_prob(xs)).mean()
loss.backward()
print("Gradient of mu is:", mu.grad)
print("Gradient of sigma is:", sigma.grad)