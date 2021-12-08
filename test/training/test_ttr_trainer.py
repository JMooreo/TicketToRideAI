# import copy
# import os
# import time
# import unittest
# import numpy as np
# from datetime import datetime
#
# from src.game.Game import Game
# from src.game.Map import USMap
# from src.game.Player import Player
# from src.game.enums.GameState import GameState
# from src.game.enums.TurnState import TurnState
# from src.training.ActionSpace import ActionSpace
# from src.training.GameNode import TrainingNode, OpponentNode
# from src.training.Strategy import Strategy
# from src.training.Trainer import Trainer
#
#
# class TrainerTest(unittest.TestCase):
#
#     def setUp(self):
#         self.trainer = Trainer()
#         self.trainer.checkpoint_directory += "/test"
#
#     def test_init(self):
#         action_space = ActionSpace(Game([Player(), Player()], USMap()))
#
#         self.assertEqual(Strategy.random(len(action_space)).tolist(), self.trainer.strategy.tolist())
#         self.assertEqual(GameState.FIRST_ROUND, self.trainer.tree.game.state)
#         self.assertIsNotNone(self.trainer.tree)
#
#     def test_negative_training_step(self):
#         with self.assertRaises(ValueError):
#             self.trainer.train(-1)
#
#     def test_zero_training_step(self):
#         with self.assertRaises(ValueError):
#             self.trainer.train(0)
#
#     def test_one_training_step(self):
#         self.trainer.training_step()
#
#         self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
#         self.assertEqual(0, self.trainer.tree.game.current_player_index)
#         self.assertTrue(isinstance(self.trainer.tree.current_node, TrainingNode))
#
#     def test_two_training_steps(self):
#         self.trainer.training_step()
#         self.trainer.training_step()
#
#         self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.trainer.tree.game.turn_state)
#         self.assertEqual(0, self.trainer.tree.game.current_player_index)
#         self.assertTrue(isinstance(self.trainer.tree.current_node, TrainingNode))
#
#     def test_training_step_from_opponent_node(self):
#         self.trainer.tree.simulate_for_n_turns(1)
#         self.assertTrue(isinstance(self.trainer.tree.current_node, OpponentNode))
#
#         self.trainer.training_step()
#         self.assertEqual(GameState.PLAYING, self.trainer.tree.game.state)
#
#     def test_train_every_training_node_for_one_game_cycle(self):
#         self.trainer.tree.simulate_for_n_turns(90)
#         self.trainer.train(1)
#
#         self.assertEqual(GameState.FIRST_ROUND, self.trainer.tree.game.state)
#
#     def test_load_np_array(self):
#         with open(f"{self.trainer.checkpoint_directory}test_check-asdfasdf.txt", "w") as f:
#             np.savetxt(f, np.ones(141))
#
#         loaded = np.loadtxt(f"{self.trainer.checkpoint_directory}test_check-asdfasdf.txt")
#         self.assertEqual(loaded.tolist(), np.ones(141).tolist())
#
#         os.remove(f"{self.trainer.checkpoint_directory}test_check-asdfasdf.txt")
#
#     def test_load_latest_checkpoint(self):
#         try:
#             os.mkdir(self.trainer.checkpoint_directory)
#             os.chmod(self.trainer.checkpoint_directory, 0o777)
#         except:
#             pass
#
#         with open(f"{self.trainer.checkpoint_directory}/checkpoint-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.txt", "w") as f:
#             np.savetxt(f, np.ones(141))
#
#         time.sleep(1)
#
#         with open(f"{self.trainer.checkpoint_directory}/checkpoint-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.txt", "w") as f:
#             np.savetxt(f, np.zeros(141))
#
#         expected = np.zeros(141)
#         self.trainer.load_latest_checkpoint()
#         actual = self.trainer.strategy
#
#         self.assertEqual(expected.tolist(), actual.tolist())
#
#         try:
#             os.rmdir(self.trainer.checkpoint_directory)
#         except:
#             pass
#
#     def test_load_latest_checkpoint_no_checkpoints(self):
#         self.trainer.checkpoint_directory += "/directory_that_does_not_exist"
#         try:
#             os.mkdir(self.trainer.checkpoint_directory)
#         except:
#             pass
#
#         self.trainer.load_latest_checkpoint()
#
#         expected = Strategy.random(len(ActionSpace(self.trainer.tree.game))).tolist()
#         actual = self.trainer.strategy.tolist()
#
#         self.assertEqual(expected, actual)
#
#         try:
#             os.rmdir(self.trainer.checkpoint_directory)
#         except:
#             pass
#
#     def test_opponent_is_allowed_to_use_latest_checkpoint_but_doesnt_train(self):
#         self.trainer.checkpoint_directory = self.trainer.checkpoint_directory[:-5]
#         self.trainer.load_latest_checkpoint()
#
#         last_checkpoint = copy.deepcopy(self.trainer.opponent_strategy)
#         self.trainer.checkpoint_directory += "/test"
#         try:
#             os.mkdir(self.trainer.checkpoint_directory)
#         except:
#             pass
#
#         self.assertEqual(self.trainer.strategy.tolist(), self.trainer.opponent_strategy.tolist())
#
#         self.trainer.tree.simulate_for_n_turns(90)
#         self.trainer.train(1)
#
#         self.assertNotEqual(self.trainer.strategy.tolist(), self.trainer.opponent_strategy.tolist())
#         self.assertEqual(last_checkpoint.tolist(), self.trainer.opponent_strategy.tolist())
