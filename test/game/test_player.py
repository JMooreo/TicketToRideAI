import unittest

from src.game.CardList import CardList
from src.game.Player import Player


class PlayerTest(unittest.TestCase):

    def setUp(self):
        self.player = Player()

    def test_init(self):
        self.assertEqual(CardList(), self.player.hand)
        self.assertEqual([], self.player.owned_destinations)
        self.assertEqual([], self.player.turn_history)
