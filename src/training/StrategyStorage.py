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

    def get(self, destinations: Dict[int, Destination]):
        return self.__get_strategy_by_key(get_key(destinations))

    def update(self, destinations: Dict[int, Destination], regrets: ndarray):
        key = get_key(destinations)
        if key == "[]":
            raise ValueError

        current_strategy = self.strategies.get(key, Strategy.random(141))
        self.strategies[key] = current_strategy + regrets

    def __get_strategy_by_key(self, key: str):
        return self.strategies.get(key, Strategy.random(141))
