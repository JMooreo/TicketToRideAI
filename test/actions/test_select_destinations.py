import unittest

import numpy as np

from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace


class SelectDestinationsActionTest(unittest.TestCase):

    def setUp(self):
        self.players = [Player(), Player()]
        self.game = Game(self.players, USMap())
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS

    def test_init(self):
        action = SelectDestinationAction(self.game, 0)

        self.assertIs(self.game, action.game)

    def test_select_none(self):
        with self.assertRaises(ValueError):
            SelectDestinationAction(self.game, None)

    def test_destination_index_below_minimum(self):
        with self.assertRaises(IndexError):
            SelectDestinationAction(self.game, -1)

    def test_destination_index_above_maximum(self):
        with self.assertRaises(IndexError):
            SelectDestinationAction(self.game, 900)

    def test_init_valid(self):
        self.game.available_destinations = [0, 1, 2]
        self.game.state = GameState.PLAYING
        action = SelectDestinationAction(self.game, 0)

        self.assertTrue(action.is_valid())

    def test_destination_not_available(self):
        action = SelectDestinationAction(self.game, 1)
        self.game.available_destinations = []

        self.assertFalse(action.is_valid())

    def test_state_after_action(self):
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1]
        action = SelectDestinationAction(self.game, 1)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_correct_player_gets_the_destinations(self):
        self.game.available_destinations = [1, 2, 3]
        action = SelectDestinationAction(self.game, 2)

        action.execute()

        self.assertTrue(2 in self.players[0].uncompleted_destinations)

    def test_unchosen_destinations_go_back_into_the_deck(self):
        self.game.available_destinations = [2, 3, 5]
        self.game.state = GameState.PLAYING

        action = SelectDestinationAction(self.game, 2)
        self.assertTrue(action.is_valid())
        action.execute()

        action = SelectDestinationAction(self.game, 3)
        self.assertTrue(action.is_valid())
        action.execute()

        self.assertFalse(2 in self.game.unclaimed_destinations)
        self.assertFalse(3 in self.game.unclaimed_destinations)
        self.assertTrue(5 in self.game.unclaimed_destinations)
        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.game.turn_state)
        self.assertEqual([5], self.game.available_destinations)

    def test_all_game_states(self):
        self.game.available_destinations = [2, 3, 5]
        action = SelectDestinationAction(self.game, 2)

        for turn_state in TurnState:
            self.game.turn_state = turn_state

            for state in GameState:
                self.game.state = state

                if state in [GameState.FIRST_ROUND, GameState.PLAYING, GameState.LAST_ROUND] and \
                        turn_state == TurnState.SELECTING_DESTINATIONS:
                    self.assertTrue(action.is_valid(), state)
                else:
                    self.assertFalse(action.is_valid(), state)

    def test_action_space(self):
        for game_state in GameState:
            self.game.state = game_state
            for turn_state in TurnState:
                self.game.turn_state = turn_state
                expected = np.array([1 if SelectDestinationAction(self.game, destination).is_valid()
                                     else 0 for destination in self.game.map.destinations.keys()])
                actual = ActionSpace(self.game).selectable_destinations()
                self.assertTrue((expected == actual).all())
                self.assertEqual((len(self.game.map.destinations.keys()),), actual.shape)

    def test_as_string(self):
        self.assertEqual("select_dest_CALGARY_to_PHOENIX", str(SelectDestinationAction(self.game, 1)))
        self.assertEqual("select_dest_DALLAS_to_NEW_YORK", str(SelectDestinationAction(self.game, 5)))

    def test_turn_history(self):
        self.game.available_destinations = [1, 2, 3]
        player = self.game.players[self.game.current_player_index]

        self.assertEqual([], player.turn_history)

        action = SelectDestinationAction(self.game, 1)
        action.execute()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.game.turn_state)
        self.assertEqual([action], player.turn_history)

        action2 = SelectDestinationAction(self.game, 2)
        action2.execute()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.game.turn_state)
        self.assertEqual([action, action2], player.turn_history)

        action3 = SelectDestinationAction(self.game, 3)
        action3.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)
        self.assertEqual([action, action2, action3], player.turn_history)

    def test_select_a_destination_that_has_already_been_completed(self):
        self.players[0].routes = {i: USMap().routes.get(i) for i in [32, 33]}
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [6]

        action = SelectDestinationAction(self.game, 6)
        self.assertTrue(action.is_valid())
        action.execute()

        self.assertEqual(1, len(self.players[0].completed_destinations))
        self.assertEqual(0, len(self.players[0].uncompleted_destinations))

    def test_cant_select_destinations_more_than_3_times(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3, 4, 5]

        a1 = SelectDestinationAction(self.game, 1)
        a2 = SelectDestinationAction(self.game, 2)
        a3 = SelectDestinationAction(self.game, 3)
        a4 = SelectDestinationAction(self.game, 4)
        a5 = SelectDestinationAction(self.game, 5)

        self.assertTrue(a1.is_valid())
        a1.execute()

        self.assertTrue(a2.is_valid())
        a2.execute()

        self.assertTrue(a3.is_valid())
        a3.execute()

        self.assertFalse(a4.is_valid())
        self.assertFalse(a5.is_valid())
