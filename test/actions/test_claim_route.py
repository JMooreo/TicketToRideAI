import unittest

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState


class ClaimRouteActionTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.game.state = GameState.PLAYING
        self.game.turn_state = TurnState.INIT
        self.player = self.game.players[self.game.current_player_index]
        self.player.hand = CardList.from_numbers([0, 0, 0, 0, 0, 0, 0, 0, 14])

    def test_init(self):
        action = ClaimRouteAction(self.game, 3)

        self.assertIs(self.game, action.game)
        self.assertEqual(3, action.route_id)

    def test_route_id_below_minimum(self):
        with self.assertRaises(IndexError):
            ClaimRouteAction(self.game, -1)

    def test_route_id_minimum(self):
        action = ClaimRouteAction(self.game, 0)

        self.assertTrue(action.is_valid())

    def test_route_id_maximum(self):
        action = ClaimRouteAction(self.game, len(self.game.map.routes)-1)

        self.assertTrue(action.is_valid())

    def test_route_id_above_maximum(self):
        with self.assertRaises(IndexError):
            ClaimRouteAction(self.game, len(self.game.map.routes))

    def test_route_id_is_not_None(self):
        with self.assertRaises(ValueError):
            ClaimRouteAction(self.game, None)

    def test_route_id_is_not_a_string(self):
        with self.assertRaises(ValueError):
            ClaimRouteAction(self.game, "1")

    def test_cannot_claim_an_already_claimed_route(self):
        self.game.unclaimed_routes = [0, 5, 8]

        action = ClaimRouteAction(self.game, 4)

        self.assertFalse(action.is_valid())

    def test_all_game_states(self):
        for turn_state in TurnState:
            self.game.turn_state = turn_state
            action = ClaimRouteAction(self.game, 3)

            for state in GameState:
                self.game.state = state

                if state in [GameState.PLAYING, GameState.LAST_TURN] and \
                        turn_state == TurnState.INIT:
                    self.assertTrue(action.is_valid())
                else:
                    self.assertFalse(action.is_valid(), state)

    def test_player_does_not_have_the_cards_to_buy_route(self):
        god_mode_cards = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])
        route_id = 67
        action = ClaimRouteAction(self.game, route_id)
        cost = self.game.map.routes.get(route_id).cost
        self.assertEqual(CardList((TrainColor.ORANGE, 4)), cost.best_payment_option(god_mode_cards))

        self.game.players[self.game.current_player_index].hand = CardList((TrainColor.ORANGE, 3))

        self.assertFalse(action.is_valid())

    def test_player_already_owns_an_adjacent_route(self):
        action = ClaimRouteAction(self.game, 2)
        self.assertTrue(action.is_valid())

        self.game.players[self.game.current_player_index].owned_routes = [1]

        self.assertFalse(action.is_valid())

    def test_game_state_after(self):
        action = ClaimRouteAction(self.game, 2)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(GameState.PLAYING, self.game.state)

    def test_turn_state_after(self):
        action = ClaimRouteAction(self.game, 2)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(TurnState.FINISHED, self.game.turn_state)

    def test_route_added_to_player_owned_routes(self):
        action = ClaimRouteAction(self.game, 2)
        player = self.game.players[self.game.current_player_index]
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual([2], player.owned_routes)

    def test_existing_routes_are_not_destroyed(self):
        action = ClaimRouteAction(self.game, 2)
        player = self.game.players[self.game.current_player_index]
        player.owned_routes = [4]
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual([4, 2], player.owned_routes)

    def test_player_points_are_updated_appropriately(self):
        action = ClaimRouteAction(self.game, 2)

        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(self.game.map.routes.get(2).points, self.player.points)

    def test_route_is_removed_from_unclaimed_routes(self):
        action = ClaimRouteAction(self.game, 2)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertFalse(2 in self.game.unclaimed_routes)

    def test_cant_claim_the_same_route_twice(self):
        action = ClaimRouteAction(self.game, 2)
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertFalse(action.is_valid())

    def test_player_pays_with_their_cards(self):
        self.player.hand = CardList.from_numbers([0, 1, 2, 3, 4, 5, 6, 7, 8])
        action = ClaimRouteAction(self.game, 2)
        self.assertTrue(action.is_valid())
        self.assertEqual(CardList.from_numbers([0, 1, 0, 0, 0, 0, 0, 0]),
                         self.game.map.routes.get(2).cost.best_payment_option(self.player.hand))

        action.execute()

        self.assertEqual(CardList.from_numbers([0, 0, 2, 3, 4, 5, 6, 7, 8]), self.player.hand)

    def test_player_pays_with_trains(self):
        action = ClaimRouteAction(self.game, 2)
        amount_of_trains = self.game.map.routes.get(2).cost.amount
        self.assertTrue(action.is_valid())

        action.execute()

        self.assertEqual(45 - amount_of_trains, self.player.trains)
