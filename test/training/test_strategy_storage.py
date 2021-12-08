import unittest
from typing import Dict

import numpy as np

from src.game.Destination import Destination
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.ActionSpace import ActionSpace
from src.training.Strategy import Strategy
from src.training.StrategyStorage import StrategyStorage


class StrategyStorageTest(unittest.TestCase):
    def setUp(self):
        self.storage = StrategyStorage()
        self.destinations = USMap().destinations
        self.game = Game([Player(), Player()], USMap())
        self.length_AS = len(ActionSpace(self.game))
        self.random_strategy = Strategy.random(self.length_AS).tolist()

    def test_init(self):
        self.assertIsNotNone(self.storage)
        self.assertEqual({}, self.storage.strategies)

    def test_get_strategy_doesnt_exist_is_random(self):
        actual_strategy = self.storage.get({}).tolist()

        self.assertEqual(self.random_strategy, actual_strategy)

    def test_update_strategy_with_empty_destinations(self):
        with self.assertRaises(ValueError):
            self.storage.update({}, np.zeros(self.length_AS))

    def test_update_strategy_with_one_destination_no_regrets(self):
        destinations: Dict[int, Destination] = {1: self.destinations.get(1)}
        self.storage.update(destinations, np.zeros(self.length_AS))

        self.assertEqual(self.random_strategy, self.storage.get(destinations).tolist())

    def test_update_strategy_with_two_destinations_no_regrets(self):
        destinations: Dict[int, Destination] = {i: self.destinations.get(i) for i in [1, 2]}
        self.storage.update(destinations, np.zeros(self.length_AS))

        self.assertEqual(self.random_strategy, self.storage.get(destinations).tolist())

    def test_update_strategy_with_one_destination_with_regrets(self):
        destinations: Dict[int, Destination] = {1: self.destinations.get(1)}
        self.storage.update(destinations, np.arange(0, self.length_AS))

        self.assertEqual(np.arange(1, self.length_AS + 1).tolist(), self.storage.get(destinations).tolist())

    def test_get_two_destinations_order_doesnt_matter(self):
        destinations: Dict[int, Destination] = {i: self.destinations.get(i) for i in [1, 2]}
        self.storage.update(destinations, np.array([1 if i in [2, 3, 4, 5, 6] else 0 for i in range(self.length_AS)]))

        destinations_reversed: Dict[int, Destination] = {i: self.destinations.get(i) for i in [2, 1]}

        expected = self.storage.get(destinations).tolist()
        actual = self.storage.get(destinations_reversed).tolist()

        self.assertEqual(expected, actual)
        self.assertNotEqual(self.random_strategy, actual)

    def test_updates_from_reversed_destinations_still_works(self):
        destinations: Dict[int, Destination] = {i: self.destinations.get(i) for i in [1, 2]}
        self.storage.update(destinations, np.arange(0, self.length_AS))

        destinations_reversed: Dict[int, Destination] = {i: self.destinations.get(i) for i in [2, 1]}
        self.storage.update(destinations_reversed, np.arange(0, self.length_AS))

        expected = self.storage.get(destinations).tolist()
        actual = self.storage.get(destinations_reversed).tolist()

        self.assertEqual(expected, actual)
        self.assertEqual(np.array([2*i + 1 for i in range(self.length_AS)]).tolist(), actual)
        self.assertNotEqual(self.random_strategy, actual)
