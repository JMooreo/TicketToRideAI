from typing import List

from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class SelectDestinationsAction(Action):
    def __init__(self, game, destination_ids):
        for i in destination_ids:
            if i < 0 or i > len(game.map.destinations):
                raise IndexError

        super().__init__(game)
        self.selected_ids = destination_ids

    def is_valid(self):
        return 0 < len(self.selected_ids) < 4 and \
            self.game.destinations_are_available(self.selected_ids) and \
            self.game.state in [GameState.FIRST_TURN, GameState.PLAYING, GameState.LAST_TURN] and \
            self.game.turn_state == TurnState.SELECTING_DESTINATIONS

    def execute(self):
        self.game.take_destinations(self.selected_ids)
        self.game.turn_state = TurnState.FINISHED
