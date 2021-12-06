import copy

from src.game import Map
from typing import List

from src.game.CardList import CardList
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState


class Game:
    def __init__(self, players: List[Player], game_map: Map):
        self.last_turn_count = None
        if not players or not game_map:
            raise ValueError

        self.map = game_map
        self.players = players
        self.state = GameState.FIRST_ROUND
        self.turn_state = TurnState.INIT
        self.unclaimed_routes = copy.deepcopy(game_map.routes)
        self.unclaimed_destinations = copy.deepcopy(game_map.destinations)
        self.available_destinations = []
        self.deck = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])
        self.visible_cards = CardList()
        self.current_player_index = 0
        self.turn_count = 0
        self.last_turn_count = 1000

        for player in players:
            player.hand += self.deck.get_random(4)

        self.visible_cards += self.deck.get_random(5)

    def __str__(self):
        message = "Players:\n"
        for player in self.players:
            message += str(player) + "\n"

        message += f"Current Player Index: {self.current_player_index}\n" + \
                f"Game state: {self.state}\n" + \
                f"Turn state: {self.turn_state}\n" + \
                f"Unclaimed Routes: {self.unclaimed_routes}\n" + \
                f"Unclaimed Destinations: {self.unclaimed_destinations}\n" + \
                f"Available Destinations: {self.available_destinations}\n" + \
                f"Deck: {self.deck}\n" + \
                f"Visible Cards: {self.visible_cards}\n" + \
                f"Turn Count: {self.turn_count}\n"

        return message
                
    def take_destinations(self, destination_ids):
        for i in destination_ids:
            self.unclaimed_destinations.pop(i)
            self.players[self.current_player_index].destinations.append(i)

        self.available_destinations = []

    def take_card(self, color: TrainColor):
        card = CardList((color, 1))
        self.visible_cards -= card
        self.players[self.current_player_index].hand += card
        self.visible_cards += self.deck.get_random(1)

    def take_random(self):
        card = self.deck.get_random(1)
        self.players[self.current_player_index].hand += card
