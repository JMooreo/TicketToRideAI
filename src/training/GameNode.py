from abc import ABC, abstractmethod

from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class GameNode(ABC):
    def __init__(self, game):
        self.game = game
        self.game.turn_state = TurnState.INIT

    def do_game_over(self):
        self.game.state = GameState.GAME_OVER
        self.game.calculate_final_scores()

    @abstractmethod
    def next(self, action: Action):
        pass


class TrainingNode(GameNode):
    def __init__(self, game):
        super().__init__(game)
        self.game.current_player_index = 0

    def next(self, action: Action):
        action.execute()

        if self.game.turn_state == TurnState.FINISHED:
            self.game.players[0].turn_history = []

            if self.game.state == GameState.PLAYING and any([player.trains < 3 for player in self.game.players]):
                self.game.state = GameState.LAST_ROUND
                self.game.last_turn_count = self.game.turn_count + 2
            elif self.game.state == GameState.LAST_ROUND and self.game.turn_count == self.game.last_turn_count:
                self.do_game_over()
                return

            self.game.turn_count += 1
            return OpponentNode(self.game)

        return self


class OpponentNode(GameNode):
    def __init__(self, game):
        super().__init__(game)
        self.game.current_player_index = 1

    def next(self, action: Action):
        action.execute()

        if self.game.turn_state == TurnState.FINISHED:
            self.game.players[1].turn_history = []

            if self.game.state == GameState.FIRST_ROUND:
                self.game.state = GameState.PLAYING
            elif self.game.state == GameState.PLAYING and any([player.trains < 3 for player in self.game.players]):
                self.game.state = GameState.LAST_ROUND
                self.game.last_turn_count = self.game.turn_count + 2
            elif self.game.state == GameState.LAST_ROUND and self.game.turn_count == self.game.last_turn_count:
                self.do_game_over()
                return

            self.game.turn_count += 1
            return TrainingNode(self.game)

        return self
