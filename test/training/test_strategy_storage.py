import unittest
import numpy as np
from src.training.Strategy import Strategy
from src.training.StrategyStorage import StrategyStorage


class StrategyStorageTest(unittest.TestCase):
    def setUp(self):
        self.storage = StrategyStorage()

    def test_init(self):
        self.assertIsNotNone(self.storage)
        self.assertEqual({}, self.storage.node_strategies)

    def test_get_key_doesnt_exist(self):
        self.assertEqual(Strategy.random(141).tolist(), self.storage.get_node_strategy("asdfasdfsfawefa").tolist())

    def test_increment_average_strategy(self):
        action_id = 4
        player_idx = 0
        self.storage.increment_average_strategy(player_idx, action_id)

        expected = Strategy.random(141)
        expected[action_id] += 1

        self.assertEqual(expected.tolist(), self.storage.get_average_strategy(player_idx).tolist())

    def test_action_id_too_low(self):
        action_id = -1
        player_idx = 0

        with self.assertRaises(ValueError):
            self.storage.increment_average_strategy(player_idx, action_id)

    def test_action_id_too_high(self):
        action_id = 141
        player_idx = 0

        with self.assertRaises(ValueError):
            self.storage.increment_average_strategy(player_idx, action_id)

    def test_player_idx_too_high(self):
        action_id = 10
        player_idx = 2

        with self.assertRaises(ValueError):
            self.storage.increment_average_strategy(player_idx, action_id)

    def test_set_strategy(self):
        strategy = np.arange(141)

        self.storage.set("asdf", strategy)
        self.assertEqual(np.arange(141).tolist(), self.storage.get_node_strategy("asdf").tolist())
