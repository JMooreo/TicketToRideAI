import unittest

import numpy as np

from src.actions.SelectDestinationsAction import SelectDestinationsAction
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
        self.game.state = GameState.FIRST_TURN
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS

    def test_init(self):
        action = SelectDestinationsAction(self.game, [0, 1])

        self.assertIs(self.game, action.game)

    def test_length_of_selections_too_long(self):
        action = SelectDestinationsAction(self.game, [0, 1, 2, 3])

        self.assertFalse(action.is_valid())

    def test_length_of_selections_too_short(self):
        action = SelectDestinationsAction(self.game, [])

        self.assertFalse(action.is_valid())

    def test_one_destination_index_below_minimum(self):
        with self.assertRaises(IndexError):
            SelectDestinationsAction(self.game, [-1])

    def test_many_destinations_index_below_minimum(self):
        with self.assertRaises(IndexError):
            SelectDestinationsAction(self.game, [0, 4, -7])

    def test_one_destination_index_above_maximum(self):
        with self.assertRaises(IndexError):
            SelectDestinationsAction(self.game, [-1])

    def test_many_destination_index_above_maximum(self):
        with self.assertRaises(IndexError):
            SelectDestinationsAction(self.game, [6, 190, 8])

    def test_init_valid(self):
        self.game.available_destinations = [0, 1, 2]
        self.game.state = GameState.PLAYING
        action = SelectDestinationsAction(self.game, [0, 1])

        self.assertTrue(action.is_valid())

    def test_destination_not_available(self):
        action = SelectDestinationsAction(self.game, [1])
        self.game.available_destinations = []

        self.assertFalse(action.is_valid())

    def test_select_more_destinations_than_are_available(self):
        self.game.available_destinations = [2]
        action = SelectDestinationsAction(self.game, [2, 3])

        self.assertFalse(action.is_valid())

    def test_state_after_action(self):
        action = SelectDestinationsAction(self.game, [1])

        action.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_correct_player_gets_the_destinations(self):
        action = SelectDestinationsAction(self.game, [1, 2, 3])

        action.execute()

        for i in action.selected_ids:
            self.assertTrue(i in self.players[0].owned_destinations)

    def test_unchosen_destinations_go_back_into_the_deck(self):
        self.game.available_destinations = [2, 3, 5]
        self.game.state = GameState.PLAYING
        action = SelectDestinationsAction(self.game, [2, 3])
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertFalse(2 in self.game.unclaimed_destinations)
        self.assertFalse(3 in self.game.unclaimed_destinations)
        self.assertTrue(5 in self.game.unclaimed_destinations)
        self.assertEqual([], self.game.available_destinations)

    def test_all_game_states(self):
        self.game.available_destinations = [2, 3, 5]
        action = SelectDestinationsAction(self.game, [2, 3])

        for turn_state in TurnState:
            self.game.turn_state = turn_state

            for state in GameState:
                self.game.state = state

                if state in [GameState.FIRST_TURN, GameState.PLAYING, GameState.LAST_TURN] and \
                        turn_state == TurnState.SELECTING_DESTINATIONS:
                    self.assertTrue(action.is_valid(), state)
                else:
                    self.assertFalse(action.is_valid(), state)

    def test_action_space(self):
        for game_state in GameState:
            self.game.state = game_state
            for turn_state in TurnState:
                self.game.turn_state = turn_state
                expected = np.array([1 if SelectDestinationsAction(self.game, [destination]).is_valid()
                                     else 0 for destination in self.game.map.destinations.keys()])
                actual = ActionSpace(self.game).claimable_destinations()
                self.assertTrue((expected == actual).all())
                self.assertEqual((len(self.game.map.destinations.keys()),), actual.shape)

    def test_as_string(self):
        self.assertEqual("select_dest_1_2_3", str(SelectDestinationsAction(self.game, [1, 2, 3])))
        self.assertEqual("select_dest_1_2", str(SelectDestinationsAction(self.game, [1, 2])))
        self.assertEqual("select_dest_1", str(SelectDestinationsAction(self.game, [1])))
