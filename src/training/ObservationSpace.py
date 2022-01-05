import numpy as np

from src.game.Game import Game

from src.game.enums.TrainColor import TrainColor


class ObservationSpace:
    def __init__(self, game: Game):
        self.game = game
        self.shape = (len(self), 1)

    def __len__(self):
        return len(self.to_np_array())

    def __str__(self):
        return f"Player {self.game.current_player_index+1} view:\n{self.to_np_array()}"

    # Length: 2
    # Number of destinations of each player, sorted by the view of the current player.
    def num_destinations_each_player(self):
        player1_view = [len(player.uncompleted_destinations) + len(player.completed_destinations)
                         for player in self.game.players]

        return np.array(self.__get_current_player_view(player1_view))

    # Length: 2
    # Number of trains left for each player, 0 to 45, sorted by the view of the current player.
    def num_trains_left_each_player(self):
        player1_view = [player.trains for player in self.game.players]
        return np.array(self.__get_current_player_view(player1_view))

    # Length: 2
    # Number of cards in the hand of each player, 0 to 110 for each player, sorted by the view of the current player.
    def num_cards_each_player(self):
        player1_view = [len(player.hand) for player in self.game.players]

        return np.array(self.__get_current_player_view(player1_view))

    # Length: 2
    # The current points of each player from 0 to 300, sorted by the view of the current player.
    def points_each_player(self):
        player1_view = [player.points for player in self.game.players]
        return np.array(self.__get_current_player_view(player1_view))

    # Length: 9
    # The Visible cards as a list of 0 to 5 for each train TrainColor, sorted by the view of the current player.
    def visible_cards(self):
        return np.array([self.game.visible_cards[color] for color in TrainColor])

    # Length: routes * 3 (270 ish)
    # The routes that have are available, unavailable and not claimed, claimed by us, or claimed by an opponent
    #  - available
    #  - owned by us
    #  - owned by an opponent
    def unclaimed_routes(self):
        return np.array([1 if route in self.game.unclaimed_routes else 0 for route in self.game.map.routes])

    def current_player_routes(self):
        return np.array([1 if route in self.game.current_player().routes else 0 for route in self.game.map.routes])

    def routes_owned_by_opponent(self):
        return np.array([1 if any([route in player.routes for idx, player in enumerate(self.game.players)
                                                if idx != self.game.current_player_index])
                         else 0 for route in self.game.map.routes])

    # PLAYER SPECIFIC INFORMATION
    # Our cards as a list of 0 to 12 or 0 to 14 for each TrainColor
    # - 0 to 12 for the colors
    # - 0 to 14 for the wilds
    def current_player_cards(self):
        return np.array([self.game.current_player().hand[color] for color in TrainColor])

    # The state of each of our destinations as a one-hot vectors.
    #  - uncompleted
    #  - completed
    def current_player_uncompleted_destinations(self):
        current_player = self.game.current_player()
        return np.array([1 if destination in current_player.uncompleted_destinations
                                else 0 for destination in self.game.map.destinations])

    def current_player_completed_destinations(self):
        current_player = self.game.current_player()

        return np.array([1 if destination in current_player.completed_destinations
                              else 0 for destination in self.game.map.destinations])

    def to_np_array(self):
        return np.concatenate([
            self.points_each_player(),
            self.num_trains_left_each_player(),
            self.num_cards_each_player(),
            self.num_destinations_each_player(),
            self.visible_cards(),
            self.current_player_cards(),
            self.unclaimed_routes(),
            self.current_player_routes(),
            self.routes_owned_by_opponent(),
            self.current_player_uncompleted_destinations(),
            self.current_player_completed_destinations()
        ])

    def __get_current_player_view(self, player1_view):
        if self.game.current_player_index == 0:
            return player1_view

        return list(reversed(player1_view))
