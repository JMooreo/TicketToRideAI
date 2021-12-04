import random

from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class DrawDestinationsAction(Action):
    def __str__(self):
        return "draw_dest"

    def __eq__(self, other):
        return isinstance(other, DrawDestinationsAction) and \
               self.game == other.game

    def is_valid(self):
        return len(self.game.unclaimed_destinations) > 0 and \
                self.game.state in [GameState.FIRST_TURN, GameState.PLAYING, GameState.LAST_TURN] and \
                self.game.turn_state == TurnState.INIT

    def execute(self):
        super().execute()
        sample_size = min(3, len(self.game.unclaimed_destinations))
        self.game.available_destinations = random.sample(list(self.game.unclaimed_destinations), sample_size)
        self.game.turn_state = TurnState.SELECTING_DESTINATIONS

