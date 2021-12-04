from src.actions.Action import Action
from src.game.CardList import CardList
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class DrawRandomCardAction(Action):
    def __str__(self):
        return "draw_rand"

    def __eq__(self, other):
        return isinstance(other, DrawRandomCardAction) and \
               self.game == other.game

    def is_valid(self):
        return self.game.deck != CardList() and \
                self.game.turn_state in [TurnState.INIT, TurnState.DRAWING_CARDS] and \
                self.game.state in [GameState.PLAYING, GameState.LAST_TURN]

    def execute(self):
        super().execute()
        self.game.take_random()

        if self.game.turn_state == TurnState.INIT:
            self.game.turn_state = TurnState.DRAWING_CARDS
        elif self.game.turn_state == TurnState.DRAWING_CARDS:
            self.game.turn_state = TurnState.FINISHED
