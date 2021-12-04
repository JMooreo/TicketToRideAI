from __future__ import annotations

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.GameNode import TrainingNode, OpponentNode

class GameTree:
    def __init__(self, game: Game):
        self.game = game
        self.current_node: TrainingNode | OpponentNode = TrainingNode(game)
        self.current_node.game.state = GameState.FIRST_TURN

    def next(self, action: Action):
        if not action.is_valid():
            raise ValueError(f"The action could not be executed because it was invalid.\n{action}")

        self.current_node = self.current_node.next(action)
