import copy

import numpy as np

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree


class ActionUtility:
    @staticmethod
    def of(action: Action, game: Game, strategy=None):
        tree = GameTree(game)
        if game.state != GameState.GAME_OVER:
            tree.next(action)
            tree.simulate_random_until_game_over(game, strategy)

        return game.players[0].points - game.players[1].points

    @staticmethod
    def from_all_branches(game: Game, strategy=None):
        action_space = ActionSpace(game)
        valid_action_ids = action_space.get_valid_action_ids()
        utilities = np.zeros(len(action_space))

        for _id in valid_action_ids:
            game_copy = copy.deepcopy(game)
            action = ActionSpace(game_copy).get_action_by_id(_id)
            print("Simulating Sub Game for", action)
            utilities[_id] = ActionUtility.of(action, game_copy, strategy)

        return utilities
