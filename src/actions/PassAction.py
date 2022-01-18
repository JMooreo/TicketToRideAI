from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.TurnState import TurnState


# This action exists as a manual override. Not intended for use during gameplay.
class PassAction(Action):
    def __init__(self, game: Game):
        super().__init__(game)

    def is_valid(self):
        return True

    def execute(self):
        self.game.turn_state = TurnState.FINISHED
