#! /bin/bash

# 设置强化学习配置文件
export CONFIG=reagent/gym/tests/configs/cartpole/discrete_dqn_cartpole_online.yaml
# 训练并且测试深度强化学习模型
./reagent/workflow/cli.py run reagent.gym.tests.test_gym.run_test_online_episode $CONFIG