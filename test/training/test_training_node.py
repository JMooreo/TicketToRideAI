import unittest

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.GameNode import TrainingNode


class TrainingNodeTest(unittest.TestCase):

    def setUp(self):
        self.game = Game([Player(), Player()], USMap())

    def test_init(self):
        TrainingNode(self.game)

