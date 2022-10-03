import unittest

import numpy as np

from actors.RandomAgent import RandomAgent
from src.DeepQLearning.DeepQNetwork import Network
from src.Environments.TTREnv import TTREnv
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.GameNode import Player1Node, Player2Node


class PlayerTest(unittest.TestCase):

    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.player = Player()
        self.destinations = self.game.map.destinations
        self.routes = self.game.map.routes

    def test_init(self):
        self.assertEqual(CardList(), self.player.hand)
        self.assertEqual({}, self.player.uncompleted_destinations)
        self.assertEqual({}, self.player.routes)
        self.assertEqual([], self.player.turn_history)
        self.assertEqual([], self.player.long_term_turn_history)

    def test_final_score_one_uncompleted_destination(self):
        self.player.uncompleted_destinations = {4: self.destinations.get(4)}
        expected_points = self.destinations.get(4).points * -1

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_two_uncompleted_destinations(self):
        self.player.uncompleted_destinations = {4: self.destinations.get(4), 5: self.destinations.get(5)}
        expected_points = (self.destinations.get(4).points + self.destinations.get(5).points) * -1

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_one_completed_destination(self):
        self.player.completed_destinations = {6: self.destinations.get(6)}
        self.player.routes = {32: self.routes.get(32), 33: self.routes.get(33)}
        expected_points = self.destinations.get(6).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_multiple_completed_destinations(self):
        self.player.completed_destinations = {i: self.destinations.get(i) for i in [6, 17]}
        self.player.routes = {i: self.routes.get(i) for i in [32, 33, 64, 17, 15]}

        expected_points = self.destinations.get(6).points + self.destinations.get(17).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_one_completed_and_one_failed_destination(self):
        self.player.completed_destinations = {6: self.destinations.get(6)}
        self.player.uncompleted_destinations = {17: self.destinations.get(17)}
        self.player.routes = {i: self.routes.get(i) for i in [32, 33]}

        expected_points = self.destinations.get(6).points - self.destinations.get(17).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_multiple_completed_destinations_with_starting_points(self):
        self.player.points = 20
        self.player.completed_destinations = {i: self.destinations.get(i) for i in [6, 17]}
        self.player.routes = {i: self.routes.get(i) for i in [32, 33, 64, 17, 15]}

        expected_points = self.destinations.get(6).points + self.destinations.get(17).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_points_from_one_route(self):
        self.player.routes = {i: self.routes.get(i) for i in [14]}

        expected_points = self.routes.get(14).points

        self.assertEqual(expected_points, self.player.points_from_routes())

    def test_points_from_multiple_route(self):
        self.player.routes = {i: self.routes.get(i) for i in [14, 17]}

        expected_points = self.routes.get(14).points + self.routes.get(17).points

        self.assertEqual(expected_points, self.player.points_from_routes())

    def test_update_long_term_turn_history_with_empty_turn_history(self):
        self.player.update_long_term_turn_history()

        self.assertEqual(1, len(self.player.long_term_turn_history))
        self.assertEqual(len(TTREnv().action_space), len(self.player.long_term_turn_history[0]))

    def test_long_term_turn_history_update_is_accurate(self):
        env = TTREnv()
        turn_history = []
        while isinstance(env.tree.current_node, Player1Node):
            action_id = env.action_space.sample()
            action = env.action_space.get_action_by_id(action_id)
            env.tree.next(action)
            turn_history.append(action)

        action_ids = [action.id for action in turn_history]

        expected = [1 if action_id in action_ids else 0 for action_id in range(len(env.action_space))]
        actual = env.tree.game.players[0].long_term_turn_history[0]

        self.assertEqual(expected, actual.tolist())

    def test_long_term_turn_history_length_limit(self):
        for _ in range(4):
            self.player.update_long_term_turn_history()

        self.assertEqual(len(self.player.long_term_turn_history), 3)

    def test_long_term_turn_history_rolling_accuracy(self):
        env = TTREnv()
        turn_histories = [[], []]

        for _ in range(4):
            turn_history = []

            while isinstance(env.tree.current_node, Player1Node):
                action_id = env.action_space.sample()
                action = env.action_space.get_action_by_id(action_id)
                env.tree.next(action)
                turn_history.append(action)

            turn_histories[0].append(turn_history)
            turn_history = []

            while isinstance(env.tree.current_node, Player2Node):
                action_id = env.action_space.sample()
                action = env.action_space.get_action_by_id(action_id)
                env.tree.next(action)
                turn_history.append(action)

            turn_histories[1].append(turn_history)

        tempPlayer = Player()
        for turn in turn_histories[0][-3:]:
            tempPlayer.turn_history = turn
            tempPlayer.update_long_term_turn_history()

        expected_player1 = tempPlayer.long_term_turn_history
        actual_player1 = env.tree.game.players[0].long_term_turn_history

        for expected, actual in zip(expected_player1, actual_player1):
            self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_last_turn_empty(self):
        env = TTREnv()
        expected = np.zeros(len(env.action_space))
        actual = self.player.get_last_turn()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_last_turn_one(self):
        env = TTREnv()
        action_ids = []
        while isinstance(env.tree.current_node, Player1Node):
            action_id = env.action_space.sample()
            action = env.action_space.get_action_by_id(action_id)
            env.tree.next(action)
            action_ids.append(action_id)

        expected = [1 if action_id in action_ids else 0 for action_id in range(len(env.action_space))]
        actual = env.tree.game.players[0].get_last_turn()

        self.assertEqual(expected, actual.tolist())

    def test_get_last_turn_multiple(self):
        env = TTREnv()
        env.tree.simulate_for_n_turns(6, [RandomAgent(), RandomAgent()])

        action_ids = []
        while isinstance(env.tree.current_node, Player1Node):
            action_id = env.action_space.sample()
            action = env.action_space.get_action_by_id(action_id)
            env.tree.next(action)
            action_ids.append(action_id)

        expected = [1 if action_id in action_ids else 0 for action_id in range(len(env.action_space))]
        actual = env.tree.game.players[0].get_last_turn()

        self.assertEqual(expected, actual.tolist())
