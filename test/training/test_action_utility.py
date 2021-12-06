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
        self.utility = ActionUtility(self.game)
        self.utility.game.state = GameState.LAST_ROUND
        self.utility.game.last_turn_count = self.utility.game.turn_count

    def test_init(self):
        self.game.state = GameState.FIRST_ROUND
        self.game.last_turn_count = 1000

    def test_some_amount_of_points(self):
        self.utility.game.players[0].points = 6

        self.assertEqual(6, self.utility.of(DrawRandomCardAction(self.utility.game)))

    def test_claiming_routes_doesnt_matter_when_game_is_over(self):
        self.utility.game.players[0].points = 10
        self.utility.game.state = GameState.GAME_OVER

        self.assertEqual(10, self.utility.of(ClaimRouteAction(self.utility.game, 2)))

    def test_action_gets_executed_first_by_the_simulation(self):
        self.assertEqual(1, self.utility.of(ClaimRouteAction(self.utility.game, 2)))

    def test_action_utility_of_game_already_over(self):
        self.utility.game.state = GameState.GAME_OVER
        self.utility.game.players[0].points = 80
        self.utility.game.players[1].points = 30

        self.assertEqual(50, self.utility.of(ClaimRouteAction(self.utility.game, 2)))

    def test_selecting_destinations_gives_negative_utility(self):
        self.utility.game.turn_state = TurnState.SELECTING_DESTINATIONS
        self.utility.game.available_destinations = [26]

        self.assertEqual(-20, self.utility.of(SelectDestinationAction(self.utility.game, 26)))

    def test_action_utility_from_started_game(self):
        game = Game([Player(), Player()], USMap())
        GameTree(game).simulate_for_n_turns(2)
        all_utilities = ActionUtility(game).from_all_branches()

        print("UTILITIES:")
        print(all_utilities)
        self.assertEqual(141, len(all_utilities))
        self.assertEqual(GameState.PLAYING, game.state)
