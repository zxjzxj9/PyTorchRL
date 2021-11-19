#! /usr/bin/env python

import os
import shutil
import datetime

from absl import flags
from dopamine.discrete_domains import train
import tensorflow as tf

FLAGS = flags.FLAGS

class AtariInitTest(tf.test.TestCase):

  def setUp(self):
    super(AtariInitTest, self).setUp()
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    FLAGS.gin_files = ['./dqn.gin']
    FLAGS.gin_bindings = [
        'Runner.num_iterations=1000',
        'WrappedReplayBuffer.replay_capacity = 100'  # 根据系统内存修改
    ]
    FLAGS.alsologtostderr = True

  def test_atari_init(self):
    """Tests that a DQN agent is initialized."""
    train.main([])
    shutil.rmtree(FLAGS.base_dir)

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()