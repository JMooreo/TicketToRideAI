import unittest

import gym

from src.algorithms.SingleActorDeepQLearning import SingleActorDeepQLearningAlgorithm


class SingleActorDeepQLearningTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        env = gym.make("CartPole-v0")
        algorithm = SingleActorDeepQLearningAlgorithm(env)

