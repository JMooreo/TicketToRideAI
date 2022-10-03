from abc import abstractmethod

from Environments.TTREnv import TTREnv

id_gen = (i for i in range(100000))


class Agent:
    def __init__(self):
        self.id = next(id_gen)

    def __str__(self):
        return f"{type(self).__name__}::{self.id}"

    @abstractmethod
    def act(self, env: TTREnv):
        pass
