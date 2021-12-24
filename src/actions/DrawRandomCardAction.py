from src.actions.Action import Action
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class DrawRandomCardAction(Action):
    def __init__(self, game: Game):
        super().__init__(game)
        self.name = "draw_RANDOM"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, DrawRandomCardAction) and \
               self.game == other.game

    def is_valid(self):
        return self.game.deck != CardList() and \
                self.game.turn_state in [TurnState.INIT, TurnState.DRAWING_CARDS] and \
                self.game.state in [GameState.PLAYING, GameState.LAST_ROUND]

    def execute(self):
        super().execute()
        card = self.game.deck.get_random(1)
        color = next((color for color in card.cards.keys()))
        self.name = f"draw_{color}"
        self.game.current_player().hand += card

        if self.game.turn_state == TurnState.INIT:
            self.game.turn_state = TurnState.DRAWING_CARDS
        elif self.game.turn_state == TurnState.DRAWING_CARDS:
            self.game.turn_state = TurnState.FINISHED
