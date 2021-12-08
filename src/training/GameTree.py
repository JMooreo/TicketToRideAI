from __future__ import annotations

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import TrainingNode, GameNode
from src.training.StrategyStorage import StrategyStorage


class GameTree:
    def __init__(self, game: Game):
        self.game = game
        self.current_node: GameNode = TrainingNode(game)

    def next(self, action: Action):
        if action is None or not action.is_valid():
            raise ValueError(f"The action could not be executed because it was invalid.\n" +
                             f"Action: {action}\n" +
                             str(self.game))

        self.current_node = self.current_node.next(action)

    def simulate_until_game_over(self, strategy_storage: StrategyStorage):
        action_space = ActionSpace(self.game)

        while self.game.state != GameState.GAME_OVER:
            strategy = strategy_storage.get(self.game.current_player().uncompleted_destinations)
            action, chance = action_space.get_action(strategy)
            self.next(action)

    def simulate_for_n_turns(self, num_turns, strategy_storage: StrategyStorage):
        action_space = ActionSpace(self.game)
        for _ in range(num_turns):
            if self.game.state == GameState.GAME_OVER:
                break

            node_type = self.current_node.__class__

            while isinstance(self.current_node, node_type):
                strategy = strategy_storage.get(self.game.current_player().uncompleted_destinations)
                action, chance = action_space.get_action(strategy)
                self.next(action)
