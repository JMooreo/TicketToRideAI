from typing import Dict

from numpy.core.multiarray import ndarray
import numpy as np

from src.game.Destination import Destination


# Stores Blueprint strategies for each of the destination combinations
from src.training.Strategy import Strategy


def get_key(destinations: Dict[int, Destination]):
    return str(sorted(destinations))


class StrategyStorage:
    def __init__(self):
        self.strategies = {}

    def __len__(self):
        return len(self.strategies.keys())

    def get(self, destinations: Dict[int, Destination]):
        return self.get_strategy_by_key(get_key(destinations))

    def set(self, destinations: Dict[int, Destination], new_strategy: ndarray):
        key = get_key(destinations)
        if key == "[]":
            return

        self.strategies[key] = new_strategy

    # Only intended to be used by tests
    def get_strategy_by_key(self, key: str):
        return self.strategies.get(key, Strategy.random(141))
