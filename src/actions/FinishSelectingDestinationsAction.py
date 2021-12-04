from src.actions.Action import Action
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class FinishSelectingDestinationsAction(Action):

    def __eq__(self, other):
        return isinstance(other, FinishSelectingDestinationsAction) and \
            self.game == other.game

    def is_valid(self):
        return self.game.state in [GameState.FIRST_TURN, GameState.PLAYING, GameState.LAST_TURN] and \
               self.game.turn_state == TurnState.SELECTING_DESTINATIONS and \
               self.__minimum_number_of_destinations_were_selected()

    def execute(self):
        super().execute()
        self.game.turn_state = TurnState.FINISHED

    def __minimum_number_of_destinations_were_selected(self):
        player = self.game.players[self.game.current_player_index]
        number_of_selected_destinations = sum((1 if isinstance(action, SelectDestinationAction)
                                               else 0 for action in player.turn_history))

        if self.game.state == GameState.FIRST_TURN:
            return number_of_selected_destinations > 1

        return number_of_selected_destinations > 0
