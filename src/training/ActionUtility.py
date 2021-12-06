import copy

from src.actions.Action import Action
from src.game.Game import Game
from src.training.GameTree import GameTree


class ActionUtility:
    def __init__(self, game: Game):
        self.game = copy.deepcopy(game)
        self.tree = GameTree(self.game)

    def of(self, action: Action):
        self.tree.next(action)
        self.tree.simulate_random_until_game_over()
        return self.game.players[0].points - self.game.players[1].points
