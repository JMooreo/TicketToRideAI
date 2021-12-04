import random

from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class DrawDestinationsAction(Action):
    def is_valid(self):
        return len(self.game.unclaimed_destinations) > 0 and \
               (self.game.state == GameState.FIRST_TURN or
                (self.game.state in [GameState.PLAYING, GameState.LAST_TURN]) and
                self.game.turn_state == TurnState.INIT)

    def execute(self):
        sample_size = min(3, len(self.game.unclaimed_destinations))
        self.game.available_destinations = random.sample(list(self.game.unclaimed_destinations), sample_size)
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS
