import copy

import numpy as np

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree


class ActionUtility:
    @staticmethod
    def of(action: Action, game: Game):
        print(f"Finding action utility of {action} for game:", game)
        action.game = game  # ensure this is true for safety
        tree = GameTree(game)
        if game.state != GameState.GAME_OVER:
            tree.next(action)
            tree.simulate_random_until_game_over(game)

        return game.players[0].points - game.players[1].points

    @staticmethod
    def from_all_branches(game: Game):
        print("Finding action utils for all branches for game", game)
        action_space = ActionSpace(game)
        valid_action_ids = action_space.get_valid_action_ids()
        utilities = np.zeros(len(action_space))
        print("Detected Valid Actions")
        print(valid_action_ids)

        for _id in valid_action_ids:
            game_copy = copy.deepcopy(game)
            action = ActionSpace(game_copy).get_action_by_id(_id)
            print("Simulating Sub Game for", action, "with game", game)
            print(f"Does {game} == {game_copy}?: {game == game_copy}")
            utilities[_id] = ActionUtility.of(action, game_copy)
            print("Game Info:")
            print(game_copy)

        return utilities
