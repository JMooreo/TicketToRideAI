import numpy as np
from numpy import ndarray


class Strategy:
    @classmethod
    def random(cls, size: int):
        return np.ones(size)

    @classmethod
    def normalize(cls, strategy: ndarray, _filter: ndarray = None, frequencies: ndarray = None):
        if _filter is None:
            _filter = np.ones(len(strategy))

        filtered_strategy = strategy * _filter
        normalizing_sum = sum(filtered_strategy)

        if normalizing_sum == 0:
            return np.zeros(len(strategy))

        return filtered_strategy / normalizing_sum

    @classmethod
    def from_utility(cls, strategy: ndarray, utilities: ndarray):
        return np.array([s_val if r_val <= 0
                        else r_val if s_val == 1
                        else (r_val + s_val) / 2
                        for idx, (s_val, r_val) in enumerate(zip(strategy, utilities))])

    # @classmethod
    # def from_regrets(cls, strategy: ndarray, regrets: ndarray):
    #     return np.array([s_val if r_val <= 0
    #                     else r_val if s_val == 1
    #                     else (r_val + s_val) / 2
    #                     for idx, (s_val, r_val) in enumerate(zip(strategy, regrets))])
