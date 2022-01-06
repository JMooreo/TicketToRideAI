from abc import ABC, abstractmethod


class Action(ABC):
    def __init__(self, game, action_id=-1):
        if game is None:
            raise ValueError

        self.game = game
        self.id = action_id
        self.executed = False

    def __eq__(self, other):
        return False

    def __repr__(self):
        return str(self)

    @abstractmethod
    def is_valid(self):
        pass

    def execute(self):
        if self.executed:
            raise ReferenceError("Action at this memory location already executed.")

        self.executed = True
        self.game.current_player().turn_history.append(self)
