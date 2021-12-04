from abc import ABC, abstractmethod


class Action(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def is_valid(self):
        pass

    @abstractmethod
    def execute(self):
        pass
