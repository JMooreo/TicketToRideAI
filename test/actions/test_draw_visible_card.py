import unittest

import numpy as np

from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace


class DrawVisibleCardActionTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.INIT

        for player in self.game.players:
            player.hand = CardList()

    def test_init(self):
        action = DrawVisibleCardAction(self.game, TrainColor.YELLOW)

        self.assertIs(self.game, action.game)
        self.assertEqual(TrainColor.YELLOW, action.color)

    def test_visible_cards_is_empty(self):
        self.game.visible_cards = CardList()
        action = DrawVisibleCardAction(self.game, TrainColor.YELLOW)

        self.assertFalse(action.is_valid())

    def test_one_visible_card(self):
        self.game.visible_cards = CardList((TrainColor.ORANGE, 1))
        action = DrawVisibleCardAction(self.game, TrainColor.ORANGE)

        self.assertTrue(action.is_valid())

    def test_turn_state_after(self):
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        action = DrawVisibleCardAction(self.game, TrainColor.ORANGE)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(TurnState.DRAWING_CARDS, self.game.turn_state)

    def test_visible_cards_after(self):
        self.game.deck = CardList((TrainColor.YELLOW, 1))
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        action = DrawVisibleCardAction(self.game, TrainColor.ORANGE)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertTrue(self.game.visible_cards.has(CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 3))))
        self.assertFalse(self.game.visible_cards.has(CardList((TrainColor.ORANGE, 4))))

    def test_wild_is_not_a_valid_color(self):
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        action = DrawVisibleCardAction(self.game, TrainColor.WILD)

        self.assertFalse(action.is_valid())

    def test_deck_after(self):
        self.game.deck = CardList((TrainColor.WILD, 1))
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        action = DrawVisibleCardAction(self.game, TrainColor.ORANGE)

        action.execute()

        self.assertEqual(CardList(), self.game.deck)

    def test_player_hand_after(self):
        self.game.visible_cards = CardList((TrainColor.GREEN, 1))
        action = DrawVisibleCardAction(self.game, TrainColor.GREEN)

        action.execute()

        expected = CardList((TrainColor.GREEN, 1))
        actual = self.game.players[self.game.current_player_index].hand

        self.assertEqual(expected, actual)

    def test_turn_state_after_drawing_twice(self):
        self.game.visible_cards = CardList((TrainColor.BLUE, 2))
        action = DrawVisibleCardAction(self.game, TrainColor.BLUE)

        self.assertTrue(action.is_valid())

        action.execute()
        action.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_hand_after_drawing_twice(self):
        self.game.visible_cards = CardList((TrainColor.GREEN, 1), (TrainColor.ORANGE, 1))

        actions = [
            DrawVisibleCardAction(self.game, TrainColor.GREEN),
            DrawVisibleCardAction(self.game, TrainColor.ORANGE)
        ]

        for action in actions:
            self.assertTrue(action.is_valid())
            action.execute()

        expected = CardList((TrainColor.GREEN, 1), (TrainColor.ORANGE, 1))
        actual = self.game.players[self.game.current_player_index].hand

        self.assertEqual(expected, actual)

    def test_all_game_states(self):
        self.game.visible_cards = CardList((TrainColor.YELLOW, 1))
        action = DrawVisibleCardAction(self.game, TrainColor.YELLOW)

        for turn_state in TurnState:
            self.game.turn_state = turn_state

            for state in GameState:
                self.game.state = state

                if state in [GameState.PLAYING, GameState.LAST_TURN] and \
                        turn_state in [TurnState.INIT, TurnState.DRAWING_CARDS]:
                    self.assertTrue(action.is_valid())
                else:
                    self.assertFalse(action.is_valid(), state)

    def test_action_space(self):
        for game_state in GameState:
            self.game.state = game_state
            for turn_state in TurnState:
                self.game.turn_state = turn_state
                expected = np.array([1 if DrawVisibleCardAction(self.game, color).is_valid()
                                     else 0 for color in TrainColor][:-1])

                actual = ActionSpace(self.game).drawable_visible_colored_cards()
                self.assertTrue((expected == actual).all())
                self.assertEqual((len(TrainColor)-1,), actual.shape)
