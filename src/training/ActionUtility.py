import copy

import numpy as np

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree


class ActionUtility:

    def __init__(self, game: Game):
        self.game = copy.deepcopy(game)
        self.tree = GameTree(self.game)

    def of(self, action: Action):
        if self.game.state != GameState.GAME_OVER:
            self.tree.next(action)
            self.tree.simulate_random_until_game_over(self.game)

        return self.game.players[0].points - self.game.players[1].points

    def from_all_branches(self):
        action_space = ActionSpace(self.game)
        valid_action_ids = action_space.get_valid_action_ids()
        utilities = np.zeros(len(action_space))

        for _id in valid_action_ids:
            game_copy = copy.deepcopy(self.game)
            action = ActionSpace(game_copy).get_action_by_id(_id)
            print("Simulating Sub Game for", action)
            utilities[_id] = ActionUtility(game_copy).of(action)

        return utilities
