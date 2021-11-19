#! /usr/bin/env python

import random
import pyspiel

# 初始化国际象棋强化学习环境
game = pyspiel.load_game('chess')
state = game.new_initial_state()

while not state.is_terminal():
    # 获取合法的移动，并且随机采样
    legal_actions = state.legal_actions()
    action = random.choice(legal_actions)

    # 获取合法移动的表示字符串
    action_string = state.action_to_string(state.current_player(), action)

    # 获取当前节点的状态
    print(state.is_chance_node(), 
        state.is_simultaneous_node(), 
        state.is_player_node())
    print(f"Current player: {state.current_player()}", 
        f"Action: {action_string}")

    # 执行动作
    state.apply_action(action)

# 获取最终奖励
returns = state.returns()