import numpy as np
from numpy import ndarray


class Regret:
    def __init__(self, utils: ndarray):
        if not isinstance(utils, ndarray):
            raise ValueError

        self.utils = utils

    def from_action_id(self, action_id):
        return np.array([value - self.utils[action_id] for value in self.utils])
