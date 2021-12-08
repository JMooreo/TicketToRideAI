import copy

import numpy as np

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree
from src.training.StrategyStorage import StrategyStorage


class ActionUtility:
    @staticmethod
    def of(action: Action, strategy_storage: StrategyStorage):
        game = action.game
        tree = GameTree(game)
        if game.state != GameState.GAME_OVER:
            tree.next(action)
            tree.simulate_until_game_over(strategy_storage)

        return game.players[0].points - game.players[1].points

    @staticmethod
    def from_all_branches(game: Game, strategy_storage: StrategyStorage):
        action_space = ActionSpace(game)
        valid_action_ids = action_space.get_valid_action_ids()
        utilities = np.zeros(len(action_space))

        for _id in valid_action_ids:
            game_copy = copy.deepcopy(game)
            action = ActionSpace(game_copy).get_action_by_id(_id)
            print("Simulating Sub Game for", action)
            utilities[_id] = ActionUtility.of(action, strategy_storage)

        return utilities
