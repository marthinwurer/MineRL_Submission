import logging

import gym
import gym_moving_dot
import torch
import unittest

from DQN import DQNTrainingState, DQNHyperparameters
from basic_vizdoom_env import GymBasicDoomEnv
from my_DQN import MaitlandDQN, FCDQN
from xor_env import BasicXOREnv

logger = logging.getLogger(__name__)
device = torch.device("cpu")


def setUpModule():
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDQN(unittest.TestCase):
    # def test_openai_gym(self):
    #     hyper = DQNHyperparameters(batch_size=128)
    #     env = gym.make('FrozenLake-v0')
    #     env.render("rgb_array")
    #     env.reset()
    #     logger.info("Action shape: %s" % (env.action_space.shape,))
    #     logger.info("Observation Shape: %s" % (env.observation_space.shape,))
    #
    #     state = DQNTrainingState(FCDQN, env, device, hyper)
    #     state.train_for_episodes(1)
    #     # self.assertEqual(True, False)
    def env_state(self, hyper, env, net=MaitlandDQN):
        env.render("human")
        env.reset()
        logger.info("Action shape: %s" % (env.action_space.shape,))
        logger.info("Observation Shape: %s" % (env.observation_space.shape,))

        state = DQNTrainingState(net, env, device, hyper)
        return state

    def env_tests(self, hyper, env, net=MaitlandDQN, tests=5):
        state = self.env_state(hyper, env, net)
        state.train_for_episodes(tests)
        state.run_episode(True)

    def test_basic_doom_env(self):
        hyper = DQNHyperparameters(batch_size=128)
        env = GymBasicDoomEnv(4)
        self.env_tests(hyper, env)

    def test_moving_dot(self):
        hyper = DQNHyperparameters(batch_size=128)
        env = gym.make("MovingDot-v0")
        env.max_steps = 100
        self.env_tests(hyper, env)

    def test_xor_env(self):
        hyper = DQNHyperparameters(batch_size=128)
        env = BasicXOREnv()
        self.env_tests(hyper, env, net=FCDQN, tests=500)

    def test_replay_memory_xor(self):
        hyper = DQNHyperparameters(batch_size=128)
        env = BasicXOREnv()
        state = self.env_state(hyper, env, net=FCDQN)


if __name__ == '__main__':
    unittest.main()
