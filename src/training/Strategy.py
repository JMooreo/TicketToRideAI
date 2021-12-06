import numpy as np
from numpy import ndarray


class Strategy:
    @classmethod
    def random(cls, size: int):
        return np.repeat(1, size)

    @classmethod
    def normalize(cls, strategy: ndarray, _filter: ndarray):
        filtered_strategy = strategy * _filter
        minimum_value = filtered_strategy.min()
        if minimum_value < 0:
            for idx, val in enumerate(filtered_strategy):
                filtered_strategy[idx] = val + -1 * minimum_value + 1

        normalizing_sum = sum(filtered_strategy)
        if normalizing_sum == 0:
            return np.zeros(len(strategy))

        return np.array([val/normalizing_sum for val in filtered_strategy])

    @classmethod
    def normalize_from_regrets(cls, strategy: ndarray, regrets: ndarray):
        return Strategy.normalize(strategy + regrets, np.ones(len(strategy)))
