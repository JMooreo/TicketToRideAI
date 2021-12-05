from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class SelectDestinationAction(Action):
    def __init__(self, game, destination_id):
        if not isinstance(destination_id, int):
            raise ValueError

        if destination_id < 0 or destination_id > len(game.map.destinations):
            raise IndexError

        super().__init__(game)
        self.destination_id = destination_id

    def __str__(self):
        return f"select_dest_{str(self.game.map.destinations.get(self.destination_id))}"

    def is_valid(self):
        return self.destination_id in self.game.available_destinations and \
            self.game.state in [GameState.FIRST_TURN, GameState.PLAYING, GameState.LAST_TURN] and \
            self.game.turn_state == TurnState.SELECTING_DESTINATIONS

    def execute(self):
        super().execute()

        if self.destination_id in self.game.available_destinations:
            self.game.available_destinations.remove(self.destination_id)
            self.game.unclaimed_destinations.pop(self.destination_id)
            self.game.players[self.game.current_player_index].owned_destinations.append(self.destination_id)

        if not self.game.available_destinations:
            self.game.turn_state = TurnState.FINISHED