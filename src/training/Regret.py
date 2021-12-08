import numpy as np
from numpy import ndarray


class Regret:
    def __init__(self, utils: ndarray, impact=0.005):
        if not isinstance(utils, ndarray):
            raise ValueError

        if impact <= 0 or impact > 1:
            raise ValueError(f"regret impact was weird: {impact}. Between 0 and 1 is recommended")

        self.utils = utils
        self.impact = impact

    def from_action_id(self, action_id):
        return np.array([(value - self.utils[action_id] if value != 0 else 0) * self.impact for value in self.utils])
