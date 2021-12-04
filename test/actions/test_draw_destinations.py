import unittest

from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class DrawDestinationsActionTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.INIT
        self.action = DrawDestinationsAction(self.game)

    def test_init(self):
        self.assertIs(self.game, self.action.game)

    def test_no_destinations_left(self):
        self.game.unclaimed_destinations = []

        self.assertFalse(self.action.is_valid())

    def test_one_destination_left(self):
        self.game.unclaimed_destinations = [0]

        self.assertTrue(self.action.is_valid())

    def test_two_destinations_left(self):
        self.game.unclaimed_destinations = [0, 1]

        self.assertTrue(self.action.is_valid())

    def test_three_destinations_left(self):
        self.game.unclaimed_destinations = [0, 1, 2]

        self.assertTrue(self.action.is_valid())

    def test_unclaimed_routes_after(self):
        self.game.unclaimed_destinations = [0, 1, 2, 3]
        self.assertTrue(self.action.is_valid())

        self.action.execute()

        self.assertEqual([0, 1, 2, 3], self.game.unclaimed_destinations)

    def test_available_routes_after_with_one(self):
        self.game.unclaimed_destinations = [0]
        self.assertTrue(self.action.is_valid())

        self.action.execute()

        self.assertEqual([0], self.game.available_destinations)

    def test_available_routes_after_with_two(self):
        self.game.unclaimed_destinations = [0, 1]
        self.assertTrue(self.action.is_valid())

        self.action.execute()

        self.assertEqual(2, len(self.game.available_destinations))
        for val in [0, 1]:
            self.assertTrue(val in self.game.available_destinations)

    def test_available_routes_after_with_three(self):
        self.game.unclaimed_destinations = [0, 1, 2]
        self.assertTrue(self.action.is_valid())

        self.action.execute()

        self.assertEqual(3, len(self.game.available_destinations))
        for val in [0, 1, 2]:
            self.assertTrue(val in self.game.available_destinations)

    def test_turn_state_after(self):
        self.game.unclaimed_destinations = [0, 1, 2, 3]
        self.assertTrue(self.action.is_valid())

        self.action.execute()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.game.turn_state)

    def test_all_game_states(self):
        for turn_state in TurnState:
            self.game.turn_state = turn_state

            for state in GameState:
                self.game.state = state

                if (self.game.state == GameState.FIRST_TURN or
                    (self.game.state in [GameState.PLAYING, GameState.LAST_TURN]) and
                        self.game.turn_state == TurnState.INIT):
                    self.assertTrue(self.action.is_valid())
                else:
                    self.assertFalse(self.action.is_valid(), state)
