from typing import Dict

from numpy.core.multiarray import ndarray

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
        key = get_key(destinations)

        # This helps the AI learn faster with a new set of destinations without destroying old training data
        if key not in self.strategies.keys():
            first_two_destinations = {i: destinations.get(i) for i in list(destinations.keys())[:2]}
            key = get_key(first_two_destinations)

        return self.strategies.get(key, Strategy.random(141))

    def set(self, destinations: Dict[int, Destination], new_strategy: ndarray):
        key = get_key(destinations)
        if key == "[]":
            return

        self.strategies[key] = new_strategy

    # Only intended to be used by tests
    def get_strategy_by_key(self, key: str):
        return self.strategies.get(key, Strategy.random(141))
