from abc import ABC, abstractmethod

from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class GameNode(ABC):
    def __init__(self, game):
        self.game = game

    def __do_game_over(self):
        self.game.state = GameState.GAME_OVER
        self.game.calculate_final_scores()

    def next(self, action: Action):
        action.execute()

        if self.game.turn_state == TurnState.FINISHED:
            return self.pass_turn()

        return self

    def __handle_game_state_change(self):
        if self.game.current_player_index == len(self.game.players) - 1 and self.game.state == GameState.FIRST_ROUND:
            self.game.state = GameState.PLAYING
        elif self.game.state == GameState.PLAYING and any([player.trains < 3 for player in self.game.players]):
            self.game.state = GameState.LAST_ROUND
            self.game.last_turn_count = self.game.turn_count + len(self.game.players)
        elif self.game.state == GameState.LAST_ROUND and self.game.turn_count == self.game.last_turn_count:
            self.__do_game_over()

    def pass_turn(self):
        self.game.players[self.player_index()].turn_history = []
        self.__handle_game_state_change()

        if self.game.state != GameState.GAME_OVER:
            self.game.turn_count += 1
            self.game.current_player_index = self.next_node().player_index()
            self.game.turn_state = TurnState.INIT
            return self.next_node()

    @abstractmethod
    def player_index(self):
        pass

    @abstractmethod
    def next_node(self):
        pass


class Player1Node(GameNode):
    def __init__(self, game):
        super().__init__(game)

    def player_index(self):
        return 0

    def next_node(self):
        return Player2Node(self.game)


class Player2Node(GameNode):
    def __init__(self, game):
        super().__init__(game)

    def player_index(self):
        return 1

    def next_node(self):
        return Player1Node(self.game)
