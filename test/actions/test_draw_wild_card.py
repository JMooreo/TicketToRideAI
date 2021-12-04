import unittest

from src.actions.DrawWildCardAction import DrawWildCardAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState


class DrawWildCardActionTest(unittest.TestCase):
    def setUp(self):
        self.players = [Player(), Player()]
        self.game = Game(self.players, USMap())
        self.game.turn_state = TurnState.INIT
        self.game.state = GameState.PLAYING
        self.action = DrawWildCardAction(self.game)

    def test_init(self):
        self.assertIs(self.game, self.action.game)

    def test_init_None_game(self):
        action = DrawWildCardAction(None)

        self.assertFalse(action.is_valid())

    def test_wild_is_not_available(self):
        self.game.visible_cards = CardList((TrainColor.ORANGE, 5))
        action = DrawWildCardAction(self.game)

        self.assertFalse(action.is_valid())

    def test_wild_is_available(self):
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        action = DrawWildCardAction(self.game)

        self.assertTrue(action.is_valid())

    def test_turn_state_after(self):
        action = DrawWildCardAction(self.game)
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_visible_cards_after(self):
        action = DrawWildCardAction(self.game)
        self.game.deck = CardList((TrainColor.YELLOW, 1))
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        self.assertTrue(action.is_valid())

        action.execute()

        expected_visible_cards = CardList((TrainColor.YELLOW, 1), (TrainColor.ORANGE, 4))
        self.assertEqual(expected_visible_cards, self.game.visible_cards)
        self.assertEqual(CardList(), self.game.deck)

    def test_correct_player_gets_the_card(self):
        action = DrawWildCardAction(self.game)
        self.game.deck = CardList((TrainColor.YELLOW, 1))
        self.game.visible_cards = CardList((TrainColor.WILD, 1), (TrainColor.ORANGE, 4))
        self.assertTrue(action.is_valid())

        action.execute()

        player = self.game.players[self.game.current_player_index]
        self.assertTrue(player.hand.has(CardList((TrainColor.WILD, 1))))

    def test_all_game_states(self):
        self.game.visible_cards = CardList((TrainColor.WILD, 1))

        for turn_state in TurnState:
            self.game.turn_state = turn_state

            for state in GameState:
                self.game.state = state

                if state in [GameState.PLAYING, GameState.LAST_TURN] and \
                        turn_state == TurnState.INIT:
                    self.assertTrue(self.action.is_valid())
                else:
                    self.assertFalse(self.action.is_valid(), state)

#   TODO:
#   claim route


