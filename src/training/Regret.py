import numpy as np
from numpy import ndarray


class Regret:
    def __init__(self, utils: ndarray, learning_rate=0.01):
        if not isinstance(utils, ndarray):
            raise ValueError

        if learning_rate <= 0 or learning_rate > 10:
            raise ValueError(f"learning rate was weird: {learning_rate}. Between 0 and 1 is recommended")

        self.utils = utils
        self.learning_rate = learning_rate

    def from_action_id(self, action_id):
        return np.array([(value - self.utils[action_id]) * self.learning_rate for value in self.utils])
