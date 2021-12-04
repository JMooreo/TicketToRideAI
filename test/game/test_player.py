import unittest

from src.game.CardList import CardList
from src.game.Player import Player


class PlayerTest(unittest.TestCase):

    def test_init(self):
        player = Player()

        self.assertEqual(CardList(), player.hand)
        self.assertEqual([], player.owned_destinations)
