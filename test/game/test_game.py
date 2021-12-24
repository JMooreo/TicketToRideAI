import copy
import unittest

from src.DeepQLearning.Agent import Agent
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState
from src.training.GameTree import GameTree


class GameTest(unittest.TestCase):

    def setUp(self):
        self.game = Game([Player(), Player()], USMap())

    def test_init(self):
        players = [Player(), Player()]
        my_map = USMap()
        my_game = Game(players, my_map)

        self.assertIs(players, my_game.players)
        self.assertIs(my_map, my_game.map)
        self.assertEqual(my_game.turn_state, TurnState.INIT)
        self.assertEqual(my_game.turn_count, 0)
        self.assertEqual([], my_game.available_destinations)
        self.assertEqual(GameState.FIRST_ROUND, my_game.state)
        self.assertEqual(my_game.last_turn_count, 1000)

        for route in my_map.routes.keys():
            self.assertTrue(route in my_game.unclaimed_routes)

        for destination in my_map.destinations.keys():
            self.assertTrue(destination in my_game.unclaimed_destinations)

    def test_init_no_players(self):
        with self.assertRaises(ValueError):
            my_game = Game([], USMap())

            self.assertIsNotNone(my_game.players)

    def test_init_none_players(self):
        with self.assertRaises(ValueError):
            my_game = Game(None, USMap())

            self.assertIsNone(my_game.players)

    def test_init_none_map(self):
        with self.assertRaises(ValueError):
            my_game = Game([Player()], None)

            self.assertIsNone(my_game.map)

    def test_all_valid_destinations_unclaimed_on_init(self):
        for destination in self.game.map.destinations.keys():
            self.assertTrue(destination in self.game.unclaimed_destinations)

    def test_get_multiple_valid_destinations_all_available_on_init(self):
        destinations = iter(self.game.map.destinations.keys())

        for _id in destinations:
            self.assertTrue(_id in self.game.unclaimed_destinations)

    def test_draw_card(self):
        self.game.deck = CardList((TrainColor.YELLOW, 1))
        self.game.visible_cards = CardList((TrainColor.ORANGE, 2), (TrainColor.YELLOW, 3))
        self.game.take_card(TrainColor.ORANGE)

        self.assertEqual(CardList((TrainColor.YELLOW, 4), (TrainColor.ORANGE, 1)), self.game.visible_cards)
        self.assertEqual(CardList(), self.game.deck)

    def test_players_get_cards_when_game_is_initialized(self):
        player = self.game.players[0]
        opponent = self.game.players[1]

        self.assertNotEqual(CardList(), player.hand)
        self.assertEqual(4, player.hand.number_of_cards())

        self.assertNotEqual(CardList(), opponent.hand)
        self.assertEqual(4, opponent.hand.number_of_cards())

    def test_visible_cards_are_initialized(self):
        self.assertNotEqual(CardList(), self.game.visible_cards)
        self.assertEqual(5, self.game.visible_cards.number_of_cards())

    def test_all_card_lists_add_to_the_right_amount(self):
        player = self.game.players[0]
        opponent = self.game.players[1]

        expected = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])

        self.assertEqual(expected, self.game.deck + player.hand + opponent.hand + self.game.visible_cards)

    def test_deepcopy(self):
        game = Game([Player(), Player()], USMap())
        tree = GameTree(game)
        tree.simulate_for_n_turns(3, Agent.random())

        self.assertEqual(GameState.PLAYING, game.state)
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS

        game_copy = copy.deepcopy(game)

        self.assertIsNot(game, game_copy)
        self.assertIsNot(game.players, game_copy.players)
        self.assertIsNot(game.players[0], game_copy.players[0])
        self.assertIsNot(game.players[0].hand, game_copy.players[0].hand)

        self.assertEqual(game, game_copy)
        self.assertEqual(game.turn_state, game_copy.turn_state)
        self.assertEqual(game.players[0], game_copy.players[0])
        self.assertEqual(game.players[1], game_copy.players[1])

    def test_get_current_player(self):
        player1 = Player()
        player2 = Player()
        players = [player1, player2]

        self.game.players = players

        self.game.current_player_index = 0
        self.assertEqual(player1, self.game.current_player())

        self.game.current_player_index = 1
        self.assertEqual(player2, self.game.current_player())

    def test_replenish_visible_cards_from_empty_deck(self):
        self.game.visible_cards = CardList((TrainColor.ORANGE, 4))
        self.game.deck = CardList()

        self.game.replenish_visible_cards()

        self.assertEqual(CardList((TrainColor.ORANGE, 4)), self.game.visible_cards)

    def test_game_init_static(self):
        game = Game.us_game()

        self.assertTrue(isinstance(game, Game))
        self.assertTrue(isinstance(game.map, USMap))
        self.assertTrue(isinstance(game.players[0], Player))
        self.assertTrue(isinstance(game.players[1], Player))