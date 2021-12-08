from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class SelectDestinationAction(Action):
    def __init__(self, game, destination_id):
        if not isinstance(destination_id, int):
            raise ValueError("destination Id was: ", destination_id)

        if destination_id < 0 or destination_id > len(game.map.destinations):
            raise IndexError

        super().__init__(game)
        self.destination_id = destination_id

    def __str__(self):
        destination = self.game.map.destinations.get(self.destination_id)
        return f"select_dest_{str(destination)} ({destination.points} points)"

    def __eq__(self, other):
        return isinstance(other, SelectDestinationAction) and \
                self.game == other.game

    def is_valid(self):
        return self.destination_id in self.game.available_destinations and \
               self.game.state in [GameState.FIRST_ROUND, GameState.PLAYING, GameState.LAST_ROUND] and \
               self.game.turn_state == TurnState.SELECTING_DESTINATIONS

    def execute(self):
        super().execute()
        destination = self.game.map.destinations.get(self.destination_id)
        player = self.game.current_player()

        self.game.available_destinations.remove(self.destination_id)
        self.game.unclaimed_destinations.pop(self.destination_id)

        if destination.path_from(player.routes.values()) is not None:
            player.completed_destinations[self.destination_id] = destination
        else:
            player.uncompleted_destinations[self.destination_id] = destination

        if not self.game.available_destinations:
            self.game.turn_state = TurnState.FINISHED


