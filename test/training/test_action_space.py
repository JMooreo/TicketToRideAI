import unittest

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.actions.DrawWildCardAction import DrawWildCardAction
from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.TrainColor import TrainColor
from src.training.ActionSpace import ActionSpace


class ActionSpaceTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.action_space = ActionSpace(self.game)

    def test_init(self):
        expected = (self.action_space.can_draw_random_card().shape[0] +
                    self.action_space.can_finish_selecting_destinations().shape[0] +
                    self.action_space.can_draw_wild().shape[0] +
                    self.action_space.drawable_visible_colored_cards().shape[0] +
                    self.action_space.can_draw_destinations().shape[0] +
                    self.action_space.selectable_destinations().shape[0] +
                    self.action_space.claimable_routes().shape[0],)

        self.assertEqual(expected, self.action_space.to_np_array().shape)
        self.assertEqual((141,), self.action_space.to_np_array().shape)

    def test_init_get_action(self):
        action = self.action_space.get_action()

        self.assertEqual(DrawDestinationsAction(self.game), action)

    def test_draw_destinations_action_by_id(self):
        action = self.action_space.get_action_by_id(0)

        self.assertEqual(DrawDestinationsAction(self.game), action)

    def test_finish_selecting_destinations_by_id(self):
        action = self.action_space.get_action_by_id(1)

        self.assertEqual(FinishSelectingDestinationsAction(self.game), action)

    def test_draw_random_card_action_by_id(self):
        action = self.action_space.get_action_by_id(2)

        self.assertEqual(DrawRandomCardAction(self.game), action)

    def test_draw_visible_colored_card_action_by_id(self):
        offset = 3

        for i in range(offset, offset + len(TrainColor) - 1):
            action = self.action_space.get_action_by_id(i)

            self.assertEqual(DrawVisibleCardAction(self.game, TrainColor(i-offset)), action)

    def test_draw_wild_card_action_by_id(self):
        action = self.action_space.get_action_by_id(3 + len(TrainColor)-1)

        self.assertEqual(DrawWildCardAction(self.game), action)

    def test_claim_route_action_by_id(self):
        route_ids = self.game.map.routes.keys()
        offset = 3 + len(TrainColor)
        for i in range(offset, offset + len(route_ids)):
            action = self.action_space.get_action_by_id(i)

            self.assertEqual(ClaimRouteAction(self.game, i-offset), action)
