import numpy as np

from src.game.Game import Game

from src.game.enums.TrainColor import TrainColor


class ObservationSpace:
    def __init__(self, game: Game):
        self.game = game

    # GENERAL INFORMATION

    # Number of destinations of each player
    def num_destinations_each_player(self):
        return np.array([len(player.uncompleted_destinations) + len(player.completed_destinations)
                         for player in self.game.players])

    # Number of trains left for each player, 0 to 45
    def num_trains_left_each_player(self):
        return np.array([player.trains for player in self.game.players])

    # Number of cards in the hand of each player, 0 to 110 for each player
    def num_cards_each_player(self):
        return np.array([len(player.hand) for player in self.game.players])

    # The Visible cards as a list of 0 to 5 for each train TrainColor
    def visible_cards(self):
        return np.array([self.game.visible_cards[color] for color in TrainColor])

    # The current points of each player from 0 to 300
    def points_each_player(self):
        return np.array([player.points for player in self.game.players])

    # The routes that have are available, unavailable and not claimed, claimed by us, or claimed by an opponent
    #  - 0 available
    #  - 1 owned by us
    #  - 2 owned by an opponent
    def routes(self):
        return np.array([0 if route in self.game.unclaimed_routes
                        else 1 if route in self.game.current_player().routes
                        else 2 if any([route in player.routes for idx, player in enumerate(self.game.players) if idx != self.game.current_player_index])
                        else ValueError  # Should be an unreachable state. Error
                        for route in self.game.map.routes])

    # PLAYER SPECIFIC INFORMATION
    # Our cards as a list of 0 to 12 or 0 to 14 for each TrainColor
    # - 0 to 12 for the colors
    # - 0 to 14 for the wilds
    def current_player_cards(self):
        return np.array([self.game.current_player().hand[color] for color in TrainColor])

    # The state of each of our destinations as a list of numbers from 0 to 2.
    #  - 0 not owned
    #  - 1 owned
    #  - 2 completed
    def current_player_destinations(self):
        current_player = self.game.current_player()
        return np.array([1 if destination in current_player.uncompleted_destinations
                        else 2 if destination in current_player.completed_destinations
                        else 0  # Must be in the deck or in another player's hand
                        for destination in self.game.map.destinations])

    def to_np_array(self):
        return np.concatenate([
            self.points_each_player(),
            self.num_trains_left_each_player(),
            self.num_cards_each_player(),
            self.num_destinations_each_player(),
            self.visible_cards(),
            self.current_player_cards(),
            self.routes(),
            self.current_player_destinations()
        ])


