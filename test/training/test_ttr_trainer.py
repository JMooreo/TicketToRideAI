import unittest

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.Strategy import Strategy
from src.training.Trainer import Trainer


class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()

    def test_init(self):
        action_space = ActionSpace(Game([Player(), Player()], USMap()))

        self.assertEqual(Strategy.random(len(action_space)).tolist(), self.trainer.strategy.tolist())
        self.assertEqual(GameState.FIRST_ROUND, self.trainer.tree.game.state)
        self.assertIsNotNone(self.trainer.tree)

    def test_negative_training_step(self):
        with self.assertRaises(ValueError):
            self.trainer.train(-1)

    def test_zero_training_step(self):
        with self.assertRaises(ValueError):
            self.trainer.train(0)

    def test_one_training_step(self):
        self.trainer.training_step()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)

    def test_two_training_steps(self):
        self.trainer.training_step()
        self.trainer.training_step()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)
