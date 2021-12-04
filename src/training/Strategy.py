import numpy as np
from numpy import ndarray


class Strategy:
    @classmethod
    def random(cls, size: int):
        return np.repeat(1/size, size)

    @classmethod
    def normalize(cls, strategy: ndarray, _filter: ndarray):
        filtered_strategy = strategy * _filter
        normalizing_sum = sum(filtered_strategy)
        if normalizing_sum == 0:
            return np.zeros(len(strategy))

        normalized_strategy = np.zeros(len(strategy))

        for i, val in enumerate(filtered_strategy):
            normalized_strategy[i] = val/normalizing_sum

        return normalized_strategy

