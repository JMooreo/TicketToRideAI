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

    def test_set_strategy(self):
        strategy = np.arange(141)

        self.storage.set("asdf", strategy)
        self.assertEqual(np.arange(141).tolist(), self.storage.get_node_strategy("asdf").tolist())
