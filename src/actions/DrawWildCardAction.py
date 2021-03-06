from src.actions.Action import Action
from src.game.CardList import CardList
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState


class DrawWildCardAction(Action):
    def __str__(self):
        return "draw_WILD"

    def __eq__(self, other):
        return isinstance(other, DrawWildCardAction) and \
                self.game == other.game

    def is_valid(self):
        return self.game is not None and \
               self.game.turn_state == TurnState.INIT and \
               self.game.visible_cards.has(CardList((TrainColor.WILD, 1))) and \
               self.game.state in [GameState.PLAYING, GameState.LAST_ROUND]

    def execute(self):
        super().execute()
        self.game.take_card(TrainColor.WILD)
        self.game.turn_state = TurnState.FINISHED
