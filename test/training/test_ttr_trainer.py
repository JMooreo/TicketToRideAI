import os
import pickle
import unittest

from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import Player1Node
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
        self.trainer.training_step()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)
        self.assertTrue(isinstance(self.trainer.tree.current_node, Player1Node))
        self.assertEqual(Strategy.random(len(ActionSpace(self.trainer.tree.game))).tolist(),
                         self.trainer.strategy_storage.get({}).tolist())

    def test_two_training_steps(self):
        self.trainer.training_step()
        self.trainer.training_step()

        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
        self.assertEqual(0, self.trainer.tree.game.current_player_index)
        self.assertTrue(isinstance(self.trainer.tree.current_node, Player1Node))
        self.assertEqual(Strategy.random(len(ActionSpace(self.trainer.tree.game))).tolist(),
                         self.trainer.strategy_storage.get({}).tolist())

    def test_save_checkpoint(self):
        for i in range(4):
            self.trainer.training_step()

        self.trainer.save_checkpoint("./data.pkl")

        with open("./data.pkl", "rb") as f:
            loaded_dict = pickle.load(f)
            print(loaded_dict)
            for key, strategy in loaded_dict.items():
                self.assertEqual(strategy.tolist(), self.trainer.strategy_storage.get_strategy_by_key(key).tolist())

        os.remove("./data.pkl")

    def test_train_resets_to_first_round(self):
        self.trainer.tree.simulate_for_n_turns(90, StrategyStorage())
        self.trainer.train(1)

        self.assertEqual(GameState.FIRST_ROUND, self.trainer.tree.game.state)

    def test_display_strategy(self):
        self.trainer.tree.simulate_for_n_turns(90, StrategyStorage())
        self.trainer.train(1)
        self.trainer.display_strategy()
