from typing import List
import numpy as np
from numpy import ndarray

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.actions.DrawWildCardAction import DrawWildCardAction
from src.actions.SelectDestinationsAction import SelectDestinationsAction
from src.game.enums.TrainColor import TrainColor


class ActionSpace:
    def __init__(self, game):
        self.game = game

    def __len__(self):
        return self.to_np_array().shape[0]

    # Everything that you can do given the current game state

    def can_draw_wild(self):
        return np.array([1 if DrawWildCardAction(self.game).is_valid() else 0])

    def can_draw_destinations(self):
        return np.array([1 if DrawDestinationsAction(self.game).is_valid() else 0])

    def can_draw_random_card(self):
        return np.array([1 if DrawRandomCardAction(self.game).is_valid() else 0])

    def claimable_routes(self):
        return np.array([1 if ClaimRouteAction(self.game, route).is_valid()
                         else 0 for route in self.game.map.routes.keys()])

    def drawable_visible_colored_cards(self):
        return np.array([1 if DrawVisibleCardAction(self.game, color).is_valid()
                         else 0 for color in TrainColor][:-1])

    def claimable_destinations(self):
        return np.array([1 if SelectDestinationsAction(self.game, [destination]).is_valid()
                         else 0 for destination in self.game.map.destinations.keys()])

    def to_np_array(self):
        return np.concatenate([
            self.can_draw_destinations(),
            self.can_draw_random_card(),
            self.drawable_visible_colored_cards(),
            self.can_draw_wild(),
            self.claimable_routes(),
            self.claimable_destinations(),
            ], axis=None)
