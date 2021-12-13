import os
import pickle
import unittest

from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import Player1Node, Player2Node
from src.training.GameTree import GameTree
from src.training.Strategy import Strategy
from src.training.StrategyStorage import StrategyStorage
from src.training.Trainer import Trainer


class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()
        self.trainer.checkpoint_directory = "./test_checkpoints"
        try:
            os.mkdir("./test_checkpoints")
        except:
            pass

    def tearDown(self):
        try:
            os.rmdir("./test_checkpoints")
        except:
            pass

    def test_init(self):
        self.assertEqual(GameState.FIRST_ROUND, self.trainer.tree.game.state)
        self.assertTrue(isinstance(self.trainer.tree, GameTree))
        self.assertTrue(isinstance(self.trainer.strategy_storage, StrategyStorage))

    def test_one_training_step(self):
        self.trainer.training_step(Player1Node)

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)
        self.assertTrue(isinstance(self.trainer.tree.current_node, Player1Node))
        self.assertEqual(Strategy.random(len(ActionSpace(self.trainer.tree.game))).tolist(),
                         self.trainer.strategy_storage.get_node_strategy("").tolist())

    def test_training_step_cant_take_an_action_passes_turn(self):
        self.trainer.tree.game.state = 6000
        self.trainer.training_step(Player1Node)
        self.assertTrue(isinstance(self.trainer.tree.current_node, Player2Node))
        self.trainer.training_step(Player2Node)
        self.assertTrue(isinstance(self.trainer.tree.current_node, Player1Node))

    def test_two_training_steps(self):
        self.trainer.training_step(Player1Node)
        self.trainer.training_step(Player1Node)

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)
        self.assertTrue(isinstance(self.trainer.tree.current_node, Player1Node))
        self.assertEqual(Strategy.random(len(ActionSpace(self.trainer.tree.game))).tolist(),
                         self.trainer.strategy_storage.get_node_strategy("").tolist())

    def test_save_checkpoint(self):
        for i in range(4):
            self.trainer.training_step(Player1Node)

        self.trainer.save_checkpoint("./data.pkl")

        with open("./data.pkl", "rb") as f:
            strategy_storage = pickle.load(f)
            print(strategy_storage)
            for key, strategy in strategy_storage.node_strategies.items():
                self.assertEqual(strategy.tolist(), self.trainer.strategy_storage.get_node_strategy(key).tolist())

        os.remove("./data.pkl")
