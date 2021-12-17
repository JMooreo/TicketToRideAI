from typing import Dict

from numpy.core.multiarray import ndarray

from src.game.Destination import Destination

from src.training.Strategy import Strategy


def get_key(destinations: Dict[int, Destination]):
    return str(sorted(destinations))


class StrategyStorage:
    def __init__(self):
        self.node_strategies = {}
        self.strategy_frequencies = {}

    def __len__(self):
        return len(self.node_strategies.keys())

    def get_node_strategy(self, key: str):
        return self.node_strategies.get(key, Strategy.random(141))

    def set(self, key: str, strategy: ndarray):
        if key == "":
            return

        self.node_strategies[key] = strategy

