import unittest

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionUtility import ActionUtility
from src.training.GameTree import GameTree


class ActionUtilityTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.game.state = GameState.LAST_ROUND
        self.game.last_turn_count = self.game.turn_count

    def test_init(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.last_turn_count = 1000

    def test_some_amount_of_points(self):
        self.game.players[0].points = 6

        self.assertEqual(6, ActionUtility.of(DrawRandomCardAction(self.game), self.game))

    def test_claiming_routes_doesnt_matter_when_game_is_over(self):
        self.game.players[0].points = 10
        self.game.state = GameState.GAME_OVER

        self.assertEqual(10, ActionUtility.of(ClaimRouteAction(self.game, 2), self.game))

    def test_action_gets_executed_first_by_the_simulation(self):
        self.assertEqual(1, ActionUtility.of(ClaimRouteAction(self.game, 2), self.game))

    def test_action_utility_of_game_already_over(self):
        self.game.state = GameState.GAME_OVER
        self.game.players[0].points = 80
        self.game.players[1].points = 30

        self.assertEqual(50, ActionUtility.of(ClaimRouteAction(self.game, 2), self.game))

    def test_selecting_destinations_gives_negative_utility(self):
        game = Game([Player(), Player()], USMap())
        game.state = GameState.LAST_ROUND
        game.turn_state = TurnState.SELECTING_DESTINATIONS
        game.unclaimed_destinations = {26: self.game.map.destinations.get(26)}
        game.last_turn_count = game.turn_count
        game.available_destinations = [26]

        self.assertEqual(-20, ActionUtility.of(SelectDestinationAction(game, 26), game))

    def test_action_utility_from_started_game(self):
        game = Game([Player(), Player()], USMap())
        GameTree(game).simulate_for_n_turns(2)
        all_utilities = ActionUtility.from_all_branches(game)

        print("UTILITIES:")
        print(all_utilities)
        self.assertEqual(141, len(all_utilities))
        self.assertEqual(GameState.PLAYING, game.state)
