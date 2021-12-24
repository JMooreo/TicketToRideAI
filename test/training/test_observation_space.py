import unittest

from src.DeepQLearning.Agent import Agent
from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawWildCardAction import DrawWildCardAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.TrainColor import TrainColor
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree
from src.training.ObservationSpace import ObservationSpace


class ObservationSpaceTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.obs = ObservationSpace(self.game)
        self.action_space = ActionSpace(self.game)

    def test_init(self):
        self.assertIs(self.obs.game, self.game)

    def test_num_each_player_destinations_init(self):
        self.assertEqual([0, 0], self.obs.num_destinations_each_player().tolist())

    def test_num_each_player_destinations_player_one(self):
        self.game.players[0].uncompleted_destinations = {2: USMap().destinations.get(2)}
        self.assertEqual([1, 0], self.obs.num_destinations_each_player().tolist())

    def test_num_each_player_destinations_player_two(self):
        self.game.players[1].uncompleted_destinations = {2: USMap().destinations.get(2)}
        self.assertEqual([0, 1], self.obs.num_destinations_each_player().tolist())

    def test_num_each_player_destinations_includes_completed(self):
        self.game.players[1].uncompleted_destinations = {2: USMap().destinations.get(2)}
        self.game.players[1].completed_destinations = {4: USMap().destinations.get(4)}
        self.assertEqual([0, 2], self.obs.num_destinations_each_player().tolist())

    def test_num_trains_left_each_player(self):
        self.assertEqual([45, 45], self.obs.num_trains_left_each_player().tolist())

    def test_num_cards_for_each_player(self):
        self.assertEqual([4, 4], self.obs.num_cards_each_player().tolist())

    def test_num_cards_for_each_player_after_drawing(self):
        tree = GameTree(self.game)
        tree.simulate_for_n_turns(2, Agent.random())
        tree.next(DrawRandomCardAction(self.game))

        self.assertEqual([5, 4], self.obs.num_cards_each_player().tolist())

    def test_visible_cards(self):
        self.game.visible_cards = CardList.from_numbers([1, 2, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual([1, 2, 2, 0, 0, 0, 0, 0, 0], self.obs.visible_cards().tolist())

    def test_points_each_player(self):
        self.assertEqual([0, 0], self.obs.points_each_player().tolist())

    def test_points_each_player_after_claim_route(self):
        tree = GameTree(self.game)
        tree.simulate_for_n_turns(2, Agent.random())
        tree.next(ClaimRouteAction(self.game, 2))

        self.assertEqual(1, self.game.current_player_index)
        self.assertEqual([0, 1], self.obs.points_each_player().tolist())

    def test_routes_all_available_on_start(self):
        expected = [0 for _ in range(len(USMap().routes))]
        self.assertEqual(expected, self.obs.routes().tolist())

    def test_claimed_routes_for_current_players_are_ones(self):
        for i in [3, 4, 5]:
            self.game.unclaimed_routes.pop(i)

        self.game.current_player().routes = {i: self.game.map.routes.get(i) for i in [3, 4, 5]}
        expected = [1 if route in [3, 4, 5] else 0 for route in self.game.map.routes]

        self.assertEqual(expected, self.obs.routes().tolist())

    def test_claimed_routes_for_opponent_twos(self):
        for i in [3, 4, 5]:
            self.game.unclaimed_routes.pop(i)

        self.game.players[1].routes = {i: self.game.map.routes.get(i) for i in [3, 4, 5]}
        expected = [2 if route in [3, 4, 5] else 0 for route in self.game.map.routes]

        self.assertEqual(expected, self.obs.routes().tolist())

    def test_current_player_and_opponent_routes(self):
        for i in [3, 4, 5, 6, 7, 8]:
            self.game.unclaimed_routes.pop(i)

        self.game.players[0].routes = {i: self.game.map.routes.get(i) for i in [6, 7, 8]}
        self.game.players[1].routes = {i: self.game.map.routes.get(i) for i in [3, 4, 5]}
        expected = [1 if route in [6, 7, 8] else 2 if route in [3, 4, 5] else 0 for route in self.game.map.routes]

        self.assertEqual(expected, self.obs.routes().tolist())

        tree = GameTree(self.game)
        tree.simulate_for_n_turns(2, Agent.random())
        self.game.visible_cards = CardList((TrainColor.WILD, 1))
        tree.next(DrawWildCardAction(self.game))

        self.assertEqual(1, self.game.current_player_index)

        expected = [2 if route in [6, 7, 8] else 1 if route in [3, 4, 5] else 0 for route in self.game.map.routes]
        self.assertEqual(expected, self.obs.routes().tolist())

    def test_current_player_cards(self):
        self.game.current_player().hand = CardList((TrainColor.ORANGE, 3), (TrainColor.WILD, 1))
        self.assertEqual([0, 0, 0, 3, 0, 0, 0, 0, 1], self.obs.current_player_cards().tolist())

    def test_destination_status_init(self):
        self.assertEqual([0 for _ in self.game.map.destinations], self.obs.current_player_destinations().tolist())

    def test_owned_but_uncompleted_destinations(self):
        for i in [1, 2, 3]:
            self.game.unclaimed_destinations.pop(i)

        self.game.current_player().uncompleted_destinations = {i: self.game.map.destinations.get(i) for i in [1, 2, 3]}
        expected = [1 if id_ in [1, 2, 3] else 0 for id_ in self.game.map.destinations]
        self.assertEqual(expected, self.obs.current_player_destinations().tolist())

    def test_owned_and_completed_destinations(self):
        for i in [1, 2, 3]:
            self.game.unclaimed_destinations.pop(i)

        self.game.current_player().completed_destinations = {i: self.game.map.destinations.get(i) for i in [1, 2, 3]}
        expected = [2 if id_ in [1, 2, 3] else 0 for id_ in self.game.map.destinations]
        self.assertEqual(expected, self.obs.current_player_destinations().tolist())

    def test_full_observation(self):
        tree = GameTree(self.game)
        tree.simulate_until_game_over(Agent.random())

        actual = self.obs.to_np_array()
        print(self.game)
        print(self.obs)

        # The current points of each player from 0 to 300
        self.assertEqual(self.obs.points_each_player().tolist(), actual[0:2].tolist())
        # Number of trains left for each player, 0 to 45
        self.assertEqual(self.obs.num_trains_left_each_player().tolist(), actual[2:4].tolist())
        # Number of cards in the hand of each player, 0 to 110 for each player
        self.assertEqual(self.obs.num_cards_each_player().tolist(), actual[4:6].tolist())
        # Number of destinations of each player
        self.assertEqual(self.obs.num_destinations_each_player().tolist(), actual[6:8].tolist())
        # The Visible cards as a list of 0 to 5 for each train TrainColor
        self.assertEqual(self.obs.visible_cards().tolist(), actual[8:17].tolist())
        # Current Player cards as a list of 0 to 12 or 0 to 14 for each TrainColor
        self.assertEqual(self.obs.current_player_cards().tolist(), actual[17:26].tolist())
        # The routes that have are available, unavailable and not claimed, claimed by us, or claimed by an opponent
        self.assertEqual(self.obs.routes().tolist(), actual[26:125].tolist())
        # The state of current player destinations as a list of numbers from 0 to 2.
        self.assertEqual(self.obs.current_player_destinations().tolist(), actual[125:155].tolist())

        for val in actual.tolist():
            self.assertTrue(isinstance(val, int))

    def test_points_view_is_for_the_current_player(self):
        self.game.players[0].points = 15
        self.game.players[1].points = 30

        self.game.current_player_index = 0
        self.assertEqual([15, 30], self.obs.points_each_player().tolist())

        self.game.current_player_index = 1
        self.assertEqual([30, 15], self.obs.points_each_player().tolist())

    def test_trains_view_is_for_current_player(self):
        self.game.players[0].trains = 15
        self.game.players[1].trains = 30

        self.game.current_player_index = 0
        self.assertEqual([15, 30], self.obs.num_trains_left_each_player().tolist())

        self.game.current_player_index = 1
        self.assertEqual([30, 15], self.obs.num_trains_left_each_player().tolist())

    def test_num_cards_view_is_for_current_player(self):
        self.game.players[0].hand = CardList((TrainColor.RED, 4))
        self.game.players[1].hand = CardList((TrainColor.YELLOW, 1))

        self.game.current_player_index = 0
        self.assertEqual([4, 1], self.obs.num_cards_each_player().tolist())

        self.game.current_player_index = 1
        self.assertEqual([1, 4], self.obs.num_cards_each_player().tolist())

    def test_num_destinations_view_is_for_current_player(self):
        self.game.players[0].uncompleted_destinations = {1: self.game.map.destinations.get(1)}
        self.game.players[0].completed_destinations = {2: self.game.map.destinations.get(2)}
        self.game.players[1].uncompleted_destinations = {3: self.game.map.destinations.get(3)}

        self.game.current_player_index = 0
        self.assertEqual([2, 1], self.obs.num_destinations_each_player().tolist())

        self.game.current_player_index = 1
        self.assertEqual([1, 2], self.obs.num_destinations_each_player().tolist())
