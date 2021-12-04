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
        if not players or not game_map:
            raise ValueError

        self.map = game_map
        self.players = players
        self.state = GameState.FIRST_TURN
        self.turn_state = TurnState.SELECTING_DESTINATIONS
        self.unclaimed_routes = copy.deepcopy(game_map.routes)
        self.unclaimed_destinations = copy.deepcopy(game_map.destinations)
        self.available_destinations = []
        self.deck = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])
        self.visible_cards = CardList()
        self.current_player_index = 0
        self.turn_count = 0

        for player in players:
            player.hand += self.deck.get_random(4)

        self.visible_cards += self.deck.get_random(5)

    def destinations_are_available(self, destination_ids):
        return all([i in self.available_destinations for i in destination_ids])

    def take_destinations(self, destination_ids):
        for i in destination_ids:
            self.unclaimed_destinations.pop(i)
            self.players[self.current_player_index].owned_destinations.append(i)

        self.available_destinations = []

    def take_card(self, color: TrainColor):
        card = CardList((color, 1))
        self.visible_cards -= card
        self.players[self.current_player_index].hand += card
        self.visible_cards += self.deck.get_random(1)

    def take_random(self):
        card = self.deck.get_random(1)
        self.players[self.current_player_index].hand += card
