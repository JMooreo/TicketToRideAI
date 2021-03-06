import unittest

from src.DeepQLearning.Agent import Agent
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree
from src.training.GameNode import Player1Node


class GameTreeTest(unittest.TestCase):

    def setUp(self):
        self.players = [Player(), Player()]
        self.game = Game(self.players, USMap())
        self.tree = GameTree(self.game)

    def __do_first_turn(self):
        for _ in range(2):
            self.tree.next(DrawDestinationsAction(self.game))
            for _ in range(3):
                self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

    def test_init(self):
        self.assertTrue(isinstance(self.tree.current_node, Player1Node))
        self.assertEqual(self.tree.training_node_type, Player1Node)

    def test_draw_destination_cards_until_there_are_none_left(self):
        self.__do_first_turn()
        for _ in range(30):
            draw = DrawDestinationsAction(self.game)
            if not draw.is_valid():
                break
            self.tree.next(draw)

            for _ in range(3):
                select = SelectDestinationAction(self.game, self.game.available_destinations[0])
                if not select.is_valid():
                    break
                self.tree.next(select)

        self.assertEqual(15, len(self.players[0].uncompleted_destinations))
        self.assertEqual(15, len(self.players[1].uncompleted_destinations))
        self.assertEqual([], self.game.available_destinations)

    def test_draw_random_cards_until_there_are_none_left(self):
        self.__do_first_turn()
        for _ in range(100):
            draw = DrawRandomCardAction(self.game)
            if not draw.is_valid():
                break
            self.tree.next(draw)

        self.assertEqual(53, self.players[0].hand.number_of_cards())
        self.assertEqual(52, self.players[1].hand.number_of_cards())
        self.assertEqual(110, (self.players[0].hand + self.players[1].hand + self.game.visible_cards).number_of_cards())
        self.assertEqual(CardList(), self.game.deck)

    def test_turn_history_resets_on_turn_switch(self):
        self.__do_first_turn()

        self.assertEqual([], self.players[0].turn_history)
        self.assertEqual([], self.players[1].turn_history)

    def test_last_round(self):
        action_space = ActionSpace(self.game)

        while not any([player.trains < 3 for player in self.game.players]):
            action = action_space.get_action()
            self.tree.next(action)

        self.assertEqual(GameState.LAST_ROUND, self.game.state)
        self.assertEqual(self.game.turn_count + 1, self.game.last_turn_count)

    def test_game_over(self):
        action_space = ActionSpace(self.game)

        while not any([player.trains < 3 for player in self.game.players]):
            action = action_space.get_action()
            self.tree.next(action)

        self.assertEqual(GameState.LAST_ROUND, self.game.state)
        self.tree.simulate_for_n_turns(1, Agent.random())
        self.assertEqual(GameState.LAST_ROUND, self.game.state)
        self.tree.simulate_for_n_turns(1, Agent.random())
        self.assertEqual(GameState.GAME_OVER, self.game.state)

    def test_random_simulation_state(self):
        self.tree.simulate_until_game_over(Agent.random())

        self.assertEqual(GameState.GAME_OVER, self.game.state)
        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_random_simulation_no_destinations_lost(self):
        self.tree.simulate_until_game_over(Agent.random())
        expected = len(self.game.map.destinations.keys())
        actual = len(self.players[0].uncompleted_destinations.keys()) + \
                 len(self.players[0].completed_destinations.keys()) + \
                 len(self.players[1].uncompleted_destinations.keys()) + \
                 len(self.players[1].completed_destinations.keys()) + \
                 len(self.game.unclaimed_destinations.keys())

        self.assertEqual(expected, actual)

    def test_random_simulation_no_routes_lost(self):
        self.tree.simulate_until_game_over(Agent.random())
        expected = len(self.game.map.routes.keys())
        actual = len(self.players[0].routes.keys()) + \
                 len(self.players[1].routes.keys()) + \
                 len(self.game.unclaimed_routes.keys())

        self.assertEqual(expected, actual)

    def test_random_simulation_no_train_cards_lost(self):
        self.tree.simulate_until_game_over(Agent.random())
        expected = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])
        actual = self.players[0].hand + \
                 self.players[1].hand + \
                 self.game.visible_cards + \
                 self.game.deck

        self.assertEqual(expected, actual)

    def test_player_points_are_accurate_total_scores(self):
        self.tree.simulate_until_game_over(Agent.random())
        expected = [player.points_from_routes() + player.points_from_destinations()
                    for player in self.game.players]

        actual = [player.points for player in self.game.players]

        self.assertEqual(expected, actual)

    def test_player_points_from_destinations_accurate(self):
        self.tree.simulate_until_game_over(Agent.random())
        expected = [player.points_from_destinations() for player in self.game.players]

        actual = []
        for player in self.game.players:
            completed_check = sum(destination.points if destination.path_from(player.routes.values()) is not None
                                  else -destination.points for destination in player.completed_destinations.values())

            uncompleted_check = sum(destination.points if destination.path_from(player.routes.values()) is not None
                                    else -destination.points for destination in player.uncompleted_destinations.values())

            actual.append(completed_check + uncompleted_check)

        self.assertEqual(expected, actual)

    def test_simulate_for_one_turn(self):
        self.assertEqual(0, self.tree.game.current_player_index)
        self.tree.simulate_for_n_turns(1, Agent.random())

        self.assertEqual(GameState.FIRST_ROUND, self.game.state)
        self.assertEqual(1, self.tree.game.current_player_index)

    def test_simulate_for_two_turns(self):
        self.assertEqual(0, self.tree.game.current_player_index)
        self.tree.simulate_for_n_turns(2, Agent.random())

        self.assertEqual(GameState.PLAYING, self.game.state)
        self.assertEqual(0, self.tree.game.current_player_index)