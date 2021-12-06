from abc import ABC, abstractmethod


class Action(ABC):
    def __init__(self, game):
        if game is None:
            raise ValueError

        self.game = game

    def __eq__(self, other):
        return False

    def __repr__(self):
        return str(self)

    @abstractmethod
    def is_valid(self):
        pass

    def execute(self):
        self.game.players[self.game.current_player_index].turn_history.append(self)
