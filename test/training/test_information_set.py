import unittest

from src.DeepQLearning.DeepQNetwork import Network
from src.DeepQLearning.TTREnv import TTREnv
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.GameTree import GameTree

from src.training.InformationSet import InformationSet


class InformationSetTest(unittest.TestCase):
    def setUp(self):
        self.tree = GameTree(Game([Player(), Player()], USMap()))

    def test_init(self):
        info = InformationSet.from_game(self.tree.game, 0)
        self.assertIsNotNone(info)

    def test_info_set_is_the_current_players_uncompleted_destinations(self):
        self.tree.simulate_for_n_turns(2, Network(TTREnv()))
        self.tree.game.current_player().uncompleted_destinations = {2: USMap().destinations.get(2)}
        self.tree.current_node.information_set = InformationSet.from_game(self.tree.game, 0)

        expected = str([2])
        actual = self.tree.current_node.information_set

        self.assertEqual(expected, actual)
