import unittest

import numpy as np

from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace


class DrawRandomCardActionTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.INIT

        for player in self.game.players:
            player.hand = CardList()

    def test_init(self):
        action = DrawRandomCardAction(self.game)
        self.assertIs(self.game, action.game)

    def test_deck_is_empty(self):
        self.game.deck = CardList()
        self.assertFalse(DrawRandomCardAction(self.game).is_valid())

    def test_deck_has_one_card(self):
        self.game.deck = CardList((TrainColor.WILD, 1))
        self.assertTrue(DrawRandomCardAction(self.game).is_valid())

    def test_turn_state_after(self):
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        self.assertTrue(DrawRandomCardAction(self.game).is_valid())

        DrawRandomCardAction(self.game).execute()

        self.assertEqual(TurnState.DRAWING_CARDS, self.game.turn_state)

    def test_visible_cards_after(self):
        action = DrawRandomCardAction(self.game)
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4)), self.game.visible_cards)

    def test_deck_after(self):
        self.game.deck = CardList((TrainColor.WILD, 1))
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))

        DrawRandomCardAction(self.game).execute()

        self.assertEqual(CardList(), self.game.deck)

    def test_player_hand_after(self):
        self.game.deck = CardList((TrainColor.WILD, 1))

        DrawRandomCardAction(self.game).execute()

        expected = CardList((TrainColor.WILD, 1))
        actual = self.game.players[self.game.current_player_index].hand

        self.assertEqual(expected, actual)

    def test_turn_state_after_drawing_random_twice(self):
        DrawRandomCardAction(self.game).execute()
        DrawRandomCardAction(self.game).execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_hand_after_drawing_random_twice(self):
        self.game.deck = CardList((TrainColor.GREEN, 1), (TrainColor.ORANGE, 1))

        DrawRandomCardAction(self.game).execute()
        DrawRandomCardAction(self.game).execute()

        expected = CardList((TrainColor.GREEN, 1), (TrainColor.ORANGE, 1))
        actual = self.game.players[self.game.current_player_index].hand

        self.assertEqual(expected, actual)

    def test_all_game_states(self):
        action = DrawRandomCardAction(self.game)

        for turn_state in TurnState:
            self.game.turn_state = turn_state

            for state in GameState:
                self.game.state = state

                if state in [GameState.PLAYING, GameState.LAST_ROUND] and \
                        turn_state in [TurnState.INIT, TurnState.DRAWING_CARDS]:
                    self.assertTrue(action.is_valid())
                else:
                    self.assertFalse(action.is_valid(), state)

    def test_action_space(self):
        for game_state in GameState:
            self.game.state = game_state
            for turn_state in TurnState:
                self.game.turn_state = turn_state
                expected = np.array([1 if DrawRandomCardAction(self.game).is_valid() else 0])
                actual = ActionSpace(self.game).can_draw_random_card()
                self.assertTrue((expected == actual).all())
                self.assertEqual((1,), actual.shape)

    def test_as_string(self):
        self.game.current_player_index = 0
        self.assertEqual("draw_RANDOM", str(DrawRandomCardAction(self.game)))

        self.game.current_player_index = 1
        self.assertEqual("draw_RANDOM", str(DrawRandomCardAction(self.game)))

    def test_turn_history(self):
        player = self.game.players[self.game.current_player_index]

        self.assertEqual([], player.turn_history)

        action = DrawRandomCardAction(self.game)
        action.execute()

        self.assertEqual(TurnState.DRAWING_CARDS, self.game.turn_state)
        self.assertEqual([action], player.turn_history)

        action2 = DrawRandomCardAction(self.game)
        action2.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)
        self.assertEqual([action, action2], player.turn_history)
