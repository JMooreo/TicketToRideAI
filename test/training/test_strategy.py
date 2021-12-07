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

        self.assertEqual(1, strategy[0])
        self.assertTrue(np.all(strategy == strategy[0]))

    def test_random_with_action_space(self):
        strategy = Strategy.random(len(self.action_space))

        self.assertEqual(1, strategy[0])
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

    def test_update_from_regrets_negative(self):
        strategy = np.repeat(1, 10)
        regrets = np.array([30 if i == 0 else -30 for i in range(10)])

        filtered_regrets = np.array([30 if i == 0 else 0 for i in range(10)])
        expected = Strategy.normalize(strategy + filtered_regrets, np.ones(len(strategy)))

        actual = Strategy.normalize_from_regrets(strategy, regrets)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_normalize_multiple_times_doesnt_do_anything(self):
        strategy = np.repeat(1, 10)
        n1 = Strategy.normalize(strategy, np.ones(len(strategy)))
        n2 = Strategy.normalize(n1, np.ones(len(strategy)))
        n3 = Strategy.normalize(n2, np.ones(len(strategy)))

        # Annoying Floating Point rounding error
        n1 = [round(val, 5) for val in n1]
        n3 = [round(val, 5) for val in n3]

        self.assertEqual(n1, n3)
