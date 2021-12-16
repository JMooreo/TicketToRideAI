import unittest

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player

from src.training.InformationSet import InformationSet


class InformationSetTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())

    def test_init(self):
        info = InformationSet.from_game(self.game, 0)
        self.assertIsNotNone(info)
