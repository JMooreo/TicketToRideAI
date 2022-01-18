import unittest

import numpy as np

from src.actions.PassAction import PassAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import Player2Node
from src.training.GameTree import GameTree


class PassActionTest(unittest.TestCase):
    def setUp(self):
        self.game = Game.us_game()
        self.tree = GameTree(self.game)
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.DRAWING_CARDS
        self.player = self.game.current_player()

    def test_drawing_cards_but_cant_draw_any_more(self):
        self.game.visible_cards = CardList((TrainColor.WILD, 1))
        self.game.deck = CardList()

        action_mask = ActionSpace(self.game).valid_action_mask()
        self.assertEqual(0, sum(action_mask))

        while sum(ActionSpace(self.game).valid_action_mask()) == 0:
            self.tree.next(PassAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
