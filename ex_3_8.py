#! /usr/bin/env python

import gym
import gym_tictactoe

env = gym.make('tictactoe-v0')
print(env.reset())
print(env.action_space.sample())
print(env.step(env.action_space.sample()))
print(env.render())