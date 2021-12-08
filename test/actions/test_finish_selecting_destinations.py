import unittest

import numpy as np

from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace


class FinishSelectingDestinationsActionTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.action = FinishSelectingDestinationsAction(self.game)

    def test_init(self):
        self.assertIs(self.game, self.action.game)

    def test_none_game(self):
        with self.assertRaises(ValueError):
            FinishSelectingDestinationsAction(None)

    def test_has_not_selected_any_destinations_during_first_turn(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS

        self.assertFalse(self.action.is_valid())

    def test_only_selected_one_destination_during_first_turn(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        SelectDestinationAction(self.game, 1).execute()

        self.assertFalse(self.action.is_valid())

    def test_selected_two_destinations_during_first_turn(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        SelectDestinationAction(self.game, 1).execute()
        SelectDestinationAction(self.game, 2).execute()

        self.assertTrue(self.action.is_valid())

    def test_selected_three_destinations_during_first_turn(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        SelectDestinationAction(self.game, 1).execute()
        SelectDestinationAction(self.game, 2).execute()
        SelectDestinationAction(self.game, 3).execute()

        self.assertFalse(self.action.is_valid())

    def test_has_selected_one_destination_during_first_turn(self):
        self.assertFalse(self.action.is_valid())

    def test_has_not_selected_any_destinations_during_regular_turn(self):
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        self.assertFalse(self.action.is_valid())

    def test_has_selected_one_destination_during_regular_turn(self):
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        select_action = SelectDestinationAction(self.game, 2)

        player = self.game.players[self.game.current_player_index]
        player.turn_history = [select_action]

        self.assertTrue(any((isinstance(action, SelectDestinationAction) for action in player.turn_history)))
        self.assertTrue(self.action.is_valid())

    def test_has_selected_two_destinations_during_regular_turn(self):
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1]

        player = self.game.players[self.game.current_player_index]
        player.uncompleted_destinations = [2, 3]

        select_action = SelectDestinationAction(self.game, 2)
        select_action2 = SelectDestinationAction(self.game, 3)

        player.turn_history = [select_action, select_action2]

        self.assertTrue(any((isinstance(action, SelectDestinationAction) for action in player.turn_history)))
        self.assertTrue(self.action.is_valid())

    def test_has_selected_three_destinations_during_regular_turn(self):
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        player = self.game.players[self.game.current_player_index]
        player.uncompleted_destinations = {}

        select_action = SelectDestinationAction(self.game, 1)
        select_action2 = SelectDestinationAction(self.game, 2)
        select_action3 = SelectDestinationAction(self.game, 3)

        select_action.execute()
        select_action2.execute()
        select_action3.execute()

        self.assertTrue(any((isinstance(action, SelectDestinationAction) for action in player.turn_history)))
        self.assertFalse(self.action.is_valid())

    def test_turn_state_after(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        SelectDestinationAction(self.game, 1).execute()

        self.action.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_turn_history_after(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        player = self.game.players[self.game.current_player_index]

        select = SelectDestinationAction(self.game, 1)
        select.execute()

        self.action.execute()

        self.assertEqual([select, self.action], player.turn_history)

    def test_finish_selecting_destinations_clears_available_destinations(self):
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.game.available_destinations = [1, 2, 3]

        player = self.game.players[self.game.current_player_index]

        select = SelectDestinationAction(self.game, 1)
        select.execute()

        self.action.execute()

        self.assertEqual([], self.game.available_destinations)


    def test_action_space(self):
        for game_state in GameState:
            self.game.state = game_state
            for turn_state in TurnState:
                self.game.turn_state = turn_state
                expected = np.array([1 if FinishSelectingDestinationsAction(self.game).is_valid() else 0])
                actual = ActionSpace(self.game).can_finish_selecting_destinations()
                self.assertTrue((expected == actual).all())
                self.assertEqual((1,), actual.shape)

    def test_to_string(self):
        self.assertEqual("finish_select_dest", str(FinishSelectingDestinationsAction(self.game)))
