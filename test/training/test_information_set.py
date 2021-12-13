import unittest

from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.TrainColor import TrainColor
from src.training.GameTree import GameTree
from src.training.InformationSet import InformationSet


class InformationSetTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())

    def test_init(self):
        info = InformationSet.for_opponents([])
        self.assertEqual("", info)

    def test_draw_destinations(self):
        action = DrawDestinationsAction(self.game)

        action.execute()
        info = InformationSet.for_opponents(self.game.players[0].turn_history)

        self.assertEqual("hidden_dest_selection", info)

    def test_draw_and_select_one_destination(self):
        action1 = DrawDestinationsAction(self.game)
        action2 = SelectDestinationAction(self.game, 2)

        action1.execute()
        self.game.available_destinations = [2]
        action2.execute()

        info = InformationSet.for_opponents(self.game.players[0].turn_history)

        self.assertEqual("hidden_dest_selection", info)

    def test_draw_and_select_two_destinations(self):
        action1 = DrawDestinationsAction(self.game)
        action2 = SelectDestinationAction(self.game, 2)
        action3 = SelectDestinationAction(self.game, 3)

        action1.execute()
        self.game.available_destinations = [2, 3]
        action2.execute()
        action3.execute()

        info = InformationSet.for_opponents(self.game.players[0].turn_history)

        self.assertEqual("hidden_dest_selection", info)

    def test_draw_and_select_two_destinations_then_finish_selecting(self):
        action1 = DrawDestinationsAction(self.game)
        action2 = SelectDestinationAction(self.game, 2)
        action3 = SelectDestinationAction(self.game, 3)
        action4 = FinishSelectingDestinationsAction(self.game)

        action1.execute()
        self.game.available_destinations = [2, 3]
        action2.execute()
        action3.execute()
        action4.execute()

        info = InformationSet.for_opponents(self.game.players[0].turn_history)

        self.assertEqual("hidden_dest_selection", info)
