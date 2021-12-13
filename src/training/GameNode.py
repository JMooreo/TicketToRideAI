from abc import ABC, abstractmethod

from node import GameNode

from src.actions.Action import Action
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.InformationSet import InformationSet


class GameNode(ABC):
    def __init__(self, game):
        self.cumulative_information_sets = ["", ""]
        self.current_turn_information_set = ""
        self.game = game

    @abstractmethod
    def player_index(self):
        pass

    @abstractmethod
    def next_node(self):
        pass

    def get_cumulative_information_set(self):
        if self.game.current_player_index == self.player_index():
            return self.cumulative_information_sets[self.player_index()] + self.current_turn_information_set

        return self.cumulative_information_sets[self.player_index()]

    def next(self, action: Action):
        action.execute()

        turn_history = self.game.current_player().turn_history
        self.current_turn_information_set = \
            f"p{self.player_index()+1}_{InformationSet.for_current_player(turn_history)}"

        if self.game.turn_state == TurnState.FINISHED:
            self.__update_other_players_information_sets()
            self.cumulative_information_sets[self.player_index()] += self.current_turn_information_set + " "
            self.game.players[self.player_index()].turn_history = []
            return self.pass_turn()

        return self

    def pass_turn(self):
        self.__handle_game_state_change()

        if self.game.state != GameState.GAME_OVER:
            self.game.turn_count += 1
            self.game.current_player_index = self.next_node().player_index()
            self.game.turn_state = TurnState.INIT
            next_node = self.next_node()
            next_node.cumulative_information_sets = self.cumulative_information_sets
            return next_node

    def __handle_game_state_change(self):
        if self.game.current_player_index == len(self.game.players) - 1 and self.game.state == GameState.FIRST_ROUND:
            self.game.state = GameState.PLAYING
        elif self.game.state == GameState.PLAYING and any([player.trains < 3 for player in self.game.players]):
            self.game.state = GameState.LAST_ROUND
            self.game.last_turn_count = self.game.turn_count + len(self.game.players)
        elif self.game.state == GameState.LAST_ROUND and self.game.turn_count == self.game.last_turn_count:
            self.game.state = GameState.GAME_OVER
            self.game.calculate_final_scores()

    def __update_other_players_information_sets(self):
        player_num = self.game.current_player_index + 1
        turn_history = self.game.current_player().turn_history

        # Update the cumulative information sets for the players who didn't just play
        # with the information that they should have received from this turn
        for player_idx, player in enumerate(self.game.players):
            if player_idx != self.game.current_player_index:
                self.cumulative_information_sets[player_idx] += \
                    f"p{player_num}_{InformationSet.for_opponents(turn_history)}" + " "


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
