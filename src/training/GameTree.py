from __future__ import annotations

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import TrainingNode, OpponentNode


class GameTree:
    def __init__(self, game: Game):
        self.game = game
        self.current_node: TrainingNode | OpponentNode = TrainingNode(game)
        self.current_node.game.state = GameState.FIRST_TURN

    def next(self, action: Action, chance=None):
        if self.game.turn_state == TurnState.INIT:
            if isinstance(self.current_node, TrainingNode):
                print("\nPlayer 1:")
            else:
                print("\nPlayer 2:")

        log = f"EXECUTING {action}"
        if chance is not None:
            log += f" ({round(100*chance, 2)}%)"

        print(log)
        print()
        print(self.game)

        if action is None or not action.is_valid():
            raise ValueError(f"The action could not be executed because it was invalid.\n{action}")

        self.current_node = self.current_node.next(action)
