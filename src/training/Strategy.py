import numpy as np
from numpy import ndarray


class Strategy:
    @classmethod
    def random(cls, size: int):
        return np.repeat(1, size)

    @classmethod
    def normalize(cls, strategy: ndarray, _filter: ndarray):
        filtered_strategy = strategy * _filter

        normalizing_sum = sum(filtered_strategy)
        if normalizing_sum == 0:
            return np.zeros(len(strategy))

        return np.array([val/normalizing_sum for val in filtered_strategy])

    @classmethod
    def normalize_from_regrets(cls, strategy: ndarray, regrets: ndarray):
        for idx, val in enumerate(regrets):
            if val <= 0:
                regrets[idx] = 0

        return Strategy.normalize(strategy + regrets, np.ones(len(strategy)))
