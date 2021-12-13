from typing import Dict

from numpy.core.multiarray import ndarray

from src.game.Destination import Destination

from src.training.Strategy import Strategy


def get_key(destinations: Dict[int, Destination]):
    return str(sorted(destinations))


class StrategyStorage:
    def __init__(self):
        self.node_strategies = {}
        self.average_strategies = {}

    def __len__(self):
        return len(self.node_strategies.keys())

    def get_node_strategy(self, key: str):
        return self.node_strategies.get(key, Strategy.random(141))

    def get_average_strategy(self, player_idx: int):
        if player_idx < 0 or player_idx > 1:
            raise ValueError

        return self.average_strategies.get(player_idx, Strategy.random(141))

    def increment_average_strategy(self, player_idx: int, action_id: int):
        if action_id < 0 or action_id > 140:
            raise ValueError

        current_strategy = self.get_average_strategy(player_idx)
        current_strategy[action_id] += 1
        self.average_strategies[player_idx] = current_strategy

    def set(self, key: str, strategy: ndarray):
        if key == "":
            return

        self.node_strategies[key] = strategy

