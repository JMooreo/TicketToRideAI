import unittest

import numpy as np

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.ActionSpace import ActionSpace
from src.training.Regret import Regret


class RegretTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())

    def test_init_no_utils(self):
        with self.assertRaises(ValueError):
            Regret(None)

    def test_init_list_utils_not_ndarray(self):
        with self.assertRaises(ValueError):
            my_list = [1, 2, 3, 4, 5]
            Regret(my_list)

    def test_init_list_utils_not_ndarry(self):
        utils = np.array([1, 2, 3, 4, 5])
        regret = Regret(utils)

        self.assertTrue((utils == regret.utils).all())

    def test_one_utility(self):
        action_space = ActionSpace(self.game)
        utils = np.array([67 if i == 0 else 0 for i in range(len(action_space))])

        expected = np.array([0 if i == 0 else -67 for i in range(len(action_space))])
        actual = Regret(utils).from_action_id(0)

        self.assertTrue((expected == actual).all())

    def test_multiple_utils(self):
        action_space = ActionSpace(self.game)
        utils = np.array([20 if i == 0
                          else 40 if i == 5
                            else 0 for i in range(len(action_space))])

        expected = np.array([0 if i == 0
                             else 20 if i == 5
                                else -20 for i in range(len(action_space))])

        actual = Regret(utils).from_action_id(0)
        self.assertTrue((expected == actual).all())
