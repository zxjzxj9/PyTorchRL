#! /usr/bin/env python

from pysc2.env import sc2_env
import random

# 配置智能体的强化学习环境参数
agent_interface_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
    rgb_dimensions=sc2_env.Dimensions(screen=128, minimap=64),
    action_space=sc2_env.ActionSpace.FEATURES,
    use_unit_counts=True,
    use_feature_units=True)

steps = 250
step_mul = 8

# 创建强化学习环境
with sc2_env.SC2Env(
    map_name=["Simple64", "Simple96"],
    players=[
        sc2_env.Agent(sc2_env.Race.random),
        sc2_env.Bot(sc2_env.Race.zerg, 
            sc2_env.Difficulty.easy, 
            sc2_env.BotBuild.rush)],
    agent_interface_format = agent_interface_format,
    step_mul=step_mul,
    game_steps_per_episode=steps * step_mul//3) as env:

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    obs = env.reset()
    # 随机选取动作，执行动作，获取奖励
    action = random.choice(obs.available_actions)
    obs = env.step([action])
    reward = obs.reward
    
