import unittest

from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.SelectDestinationsAction import SelectDestinationsAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.GameTree import GameTree
from src.training.GameNode import TrainingNode


class GameTreeTest(unittest.TestCase):

    def setUp(self):
        self.players = [Player(), Player()]
        self.game = Game(self.players, USMap())
        self.tree = GameTree(self.game)

    def __do_first_turn(self):
        self.tree.next(DrawDestinationsAction(self.game))
        self.tree.next(SelectDestinationsAction(self.game, self.game.available_destinations))

        self.tree.next(DrawDestinationsAction(self.game))
        self.tree.next(SelectDestinationsAction(self.game, self.game.available_destinations))

    def test_init(self):
        self.assertTrue(isinstance(self.tree.current_node, TrainingNode))

    def test_draw_destination_cards_until_there_are_none_left(self):
        self.__do_first_turn()
        for _ in range(30):
            draw = DrawDestinationsAction(self.game)
            if not draw.is_valid():
                break
            self.tree.next(draw)

            select = SelectDestinationsAction(self.game, self.game.available_destinations)
            if not select.is_valid():
                break
            self.tree.next(select)

        self.assertEqual(15, len(self.players[0].owned_destinations))
        self.assertEqual(15, len(self.players[1].owned_destinations))
        self.assertEqual([], self.game.available_destinations)

    def test_draw_random_cards_until_there_are_none_left(self):
        self.__do_first_turn()
        for _ in range(100):
            draw = DrawRandomCardAction(self.game)
            if not draw.is_valid():
                break
            self.tree.next(draw)

        self.assertEqual(53, self.players[0].hand.number_of_cards())
        self.assertEqual(52, self.players[1].hand.number_of_cards())
        self.assertEqual(110, (self.players[0].hand + self.players[1].hand + self.game.visible_cards).number_of_cards())
        self.assertEqual(CardList(), self.game.deck)
