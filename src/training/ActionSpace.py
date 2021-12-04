import numpy as np

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.actions.DrawWildCardAction import DrawWildCardAction
from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.enums.TrainColor import TrainColor
from src.training.Strategy import Strategy


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

    def can_finish_selecting_destinations(self):
        return np.array([1 if FinishSelectingDestinationsAction(self.game).is_valid() else 0])

    def can_draw_random_card(self):
        return np.array([1 if DrawRandomCardAction(self.game).is_valid() else 0])

    def claimable_routes(self):
        return np.array([1 if ClaimRouteAction(self.game, route).is_valid()
                         else 0 for route in self.game.map.routes.keys()])

    def drawable_visible_colored_cards(self):
        return np.array([1 if DrawVisibleCardAction(self.game, color).is_valid()
                         else 0 for color in TrainColor][:-1])

    def selectable_destinations(self):
        return np.array([1 if SelectDestinationAction(self.game, destination).is_valid()
                         else 0 for destination in self.game.map.destinations.keys()])

    def to_np_array(self):
        return np.concatenate([
            self.can_draw_destinations(),
            self.can_finish_selecting_destinations(),
            self.can_draw_random_card(),
            self.drawable_visible_colored_cards(),
            self.can_draw_wild(),
            self.claimable_routes(),
            self.selectable_destinations(),
            ], axis=None)

    def get_action(self, strategy=None):
        if strategy is None:
            random_strategy = Strategy.random(len(self))
            strategy = Strategy.normalize(random_strategy, self.to_np_array())

        return self.get_action_by_id(np.random.choice(len(self), p=strategy))

    def get_action_by_id(self, action_id):
        if action_id == 0:
            return DrawDestinationsAction(self.game)
        elif action_id == 1:
            return FinishSelectingDestinationsAction(self.game)
        elif action_id == 2:
            return DrawRandomCardAction(self.game)
        elif action_id < 3 + len(TrainColor) - 1:
            return DrawVisibleCardAction(self.game, TrainColor(action_id-3))
        elif action_id < 3 + len(TrainColor):
            return DrawWildCardAction(self.game)
        elif action_id < 3 + len(TrainColor) + len(self.game.map.routes.keys()):
            return ClaimRouteAction(self.game, action_id - 3 - len(TrainColor))

        return None
