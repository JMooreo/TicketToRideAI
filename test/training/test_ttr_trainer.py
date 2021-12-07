import unittest

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import TrainingNode, OpponentNode
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
        self.assertTrue(isinstance(self.trainer.tree.current_node, TrainingNode))

    def test_two_training_steps(self):
        self.trainer.training_step()
        self.trainer.training_step()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)
        self.assertTrue(isinstance(self.trainer.tree.current_node, TrainingNode))

    def test_training_step_from_opponent_node(self):
        self.trainer.tree.simulate_for_n_turns(1)

        self.trainer.training_step()

        self.assertEqual(GameState.PLAYING, self.trainer.tree.game.state)

    def test_train_every_training_node_for_one_game_cycle(self):
        self.trainer.train(1)

        self.assertEqual(GameState.GAME_OVER, self.trainer.tree.game.state)
