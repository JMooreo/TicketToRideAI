import unittest

from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.ActionSpace import ActionSpace


class ActionSpaceTest(unittest.TestCase):
    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.action_space = ActionSpace(self.game)

    def test_init(self):
        expected = (self.action_space.can_draw_random_card().shape[0] +
                    self.action_space.can_draw_wild().shape[0] +
                    self.action_space.drawable_visible_colored_cards().shape[0] +
                    self.action_space.can_draw_destinations().shape[0] +
                    self.action_space.claimable_destinations().shape[0] +
                    self.action_space.claimable_routes().shape[0],)

        self.assertEqual(expected, self.action_space.to_np_array().shape)
        self.assertEqual((140,), self.action_space.to_np_array().shape)
