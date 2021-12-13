from src.actions.Action import Action
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState


class DrawVisibleCardAction(Action):
    def __init__(self, game: Game, color: TrainColor):
        super().__init__(game)
        self.color = color

    def __str__(self):
        return f"draw_{str(self.color)}"

    def __eq__(self, other):
        return isinstance(other, DrawVisibleCardAction)

    def is_valid(self):
        return self.game.visible_cards.has(CardList((self.color, 1))) and \
               self.game.turn_state in [TurnState.INIT, TurnState.DRAWING_CARDS] and \
               self.color != TrainColor.WILD and \
               self.game.state in [GameState.PLAYING, GameState.LAST_ROUND]

    def execute(self):
        super().execute()
        self.game.take_card(self.color)

        if self.game.turn_state == TurnState.INIT:
            self.game.turn_state = TurnState.DRAWING_CARDS
        elif self.game.turn_state == TurnState.DRAWING_CARDS:
            self.game.turn_state = TurnState.FINISHED
