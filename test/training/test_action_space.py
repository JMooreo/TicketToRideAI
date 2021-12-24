import copy
import unittest

import numpy as np

from src.DeepQLearning.Agent import Agent
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
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree


class ActionSpaceTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.tree = GameTree(self.game)
        self.action_space = ActionSpace(self.game)

    def test_init(self):
        expected = (self.action_space.can_draw_random_card().shape[0] +
                    self.action_space.can_finish_selecting_destinations().shape[0] +
                    self.action_space.can_draw_wild().shape[0] +
                    self.action_space.drawable_visible_colored_cards().shape[0] +
                    self.action_space.can_draw_destinations().shape[0] +
                    self.action_space.selectable_destinations().shape[0] +
                    self.action_space.claimable_routes().shape[0],)

        self.assertEqual(expected, self.action_space.valid_action_mask().shape)
        self.assertEqual((141,), self.action_space.valid_action_mask().shape)

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

    def test_select_destination_action_by_id(self):
        dest_ids = self.game.map.destinations.keys()
        offset = 3 + len(TrainColor) + len(self.game.map.routes.keys())
        for i in range(offset, offset + len(dest_ids)):
            action = self.action_space.get_action_by_id(i)

            self.assertEqual(SelectDestinationAction(self.game, i-offset), action)

    def test_get_random_action(self):
        random_choice = np.random.choice(3, p=np.array([0.5, 0.5, 0]))
        for i in range(1000):
            self.assertTrue(random_choice in [0, 1])

    def test_get_maximum_action(self):
        action = self.action_space.get_action_by_id(140)
        next_none = self.action_space.get_action_by_id(141)

        self.assertIsNotNone(action)
        self.assertIsNone(next_none)

    def test_get_valid_action_ids_new_game(self):
        self.assertTrue((np.array([0]) == self.action_space.get_valid_action_ids()).all)

    def test_get_valid_action_ids_after_one_turn(self):
        GameTree(self.game).simulate_for_n_turns(1, Agent.random())
        self.assertTrue((np.array([0]) == self.action_space.get_valid_action_ids()).all)

    def test_get_valid_action_ids_after_two_turn(self):
        GameTree(self.game).simulate_for_n_turns(2, Agent.random())

        self.assertTrue(len(self.action_space.get_valid_action_ids()) > 1)

    def test_action_space_on_deepcopy_is_the_same(self):
        game = Game([Player(), Player()], USMap())
        tree = GameTree(game)

        tree.simulate_for_n_turns(4, Agent.random())

        game_copy = copy.deepcopy(game)

        a1_ids = ActionSpace(game).get_valid_action_ids()
        a2_ids = ActionSpace(game_copy).get_valid_action_ids()

        self.assertEqual(a1_ids.tolist(), a2_ids.tolist())

    def test_sample_action_space_gives_a_random_valid_action(self):
        while self.game.state != GameState.GAME_OVER:
            action_id = self.action_space.sample()
            action = self.action_space.get_action_by_id(action_id)
            self.tree.next(action)
