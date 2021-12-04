import unittest

import numpy as np

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.ActionSpace import ActionSpace
from src.training.Strategy import Strategy


class StrategyTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.action_space = ActionSpace(self.game)

    def test_random_small_size(self):
        strategy = Strategy.random(10)

        self.assertEqual(1 / 10, strategy[0])
        self.assertTrue(np.all(strategy == strategy[0]))

    def test_random_with_action_space(self):
        strategy = Strategy.random(len(self.action_space))

        self.assertEqual(1/len(self.action_space), strategy[0])
        self.assertTrue(np.all(strategy == strategy[0]))

    def test_normalize_with_zeros(self):
        strategy = Strategy.random(10)
        strategy = Strategy.normalize(strategy, np.zeros(10))

        self.assertTrue((np.zeros(10) == strategy).all())

    def test_normalize_with_one(self):
        strategy = Strategy.random(10)
        strategy = Strategy.normalize(strategy, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.assertTrue((np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] == strategy)).all())

    def test_normalize_with_two_ones(self):
        strategy = Strategy.random(10)
        strategy = Strategy.normalize(strategy, np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.assertTrue((np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0] == strategy)).all())

    def test_normalize_with_four_ones(self):
        strategy = Strategy.random(10)
        strategy = Strategy.normalize(strategy, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))

        self.assertTrue((np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0] == strategy)).all())

    def test_normalize_with_action_space(self):
        strategy = Strategy.random(len(self.action_space))
        strategy = Strategy.normalize(strategy, self.action_space.to_np_array())

        expected = np.array([1 if i == 0 else 0 for i in range(len(self.action_space))])

        self.assertTrue((expected == strategy).all())

    def test_normalize_two_different_shapes(self):
        with self.assertRaises(ValueError):
            strategy = Strategy.random(10)
            Strategy.normalize(strategy, np.ones(7))
