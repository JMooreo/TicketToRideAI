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
        self.state = GameState.FIRST_ROUND
        self.turn_state = TurnState.INIT
        self.unclaimed_routes = {i: val for i, val in game_map.routes.items()}
        self.unclaimed_destinations = {i: val for i, val in game_map.destinations.items()}
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
        message = ""
        for i, player in enumerate(self.players):
            message += f"\nPLAYER {i + 1}"
            message += "\n" + str(player)

        message += f"\nGAME INFORMATION\n" + \
                   f"Current Player: Player {self.current_player_index + 1}\n" + \
                   f"Game State: {self.state}\n" + \
                   f"Turn State: {self.turn_state}\n" + \
                   f"Unclaimed Routes: {self.unclaimed_routes}\n" + \
                   f"Unclaimed Destinations: {self.unclaimed_destinations}\n" + \
                   f"Available Destinations: {self.available_destinations}\n" + \
                   f"Deck: {self.deck}\n" + \
                   f"Visible Cards: {self.visible_cards}\n" + \
                   f"Turn Count: {self.turn_count}\n"

        return message

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Game) and \
               self.map == other.map and \
               self.players == other.players and \
               self.state == other.state and \
               self.turn_state == other.turn_state and \
               all([r in other.unclaimed_routes for r in self.unclaimed_routes]) and \
               all([d in other.unclaimed_destinations for d in self.unclaimed_destinations]) and \
               self.available_destinations == other.available_destinations and \
               self.deck == other.deck and \
               self.visible_cards == other.visible_cards and \
               self.current_player_index == other.current_player_index and \
               self.turn_count == other.turn_count and \
               self.last_turn_count == other.last_turn_count

    def take_destinations(self, destination_ids):
        for i in destination_ids:
            self.unclaimed_destinations.pop(i)
            self.players[self.current_player_index].uncompleted_destinations.append(i)

        self.available_destinations = []

    def take_card(self, color: TrainColor):
        card = CardList((color, 1))
        self.visible_cards -= card
        self.current_player().hand += card
        self.replenish_visible_cards()

    def replenish_visible_cards(self):
        while len(self.visible_cards) < 5 and len(self.deck) > 0:
            self.visible_cards += self.deck.get_random(1)

    def take_random(self):
        card = self.deck.get_random(1)
        self.players[self.current_player_index].hand += card

    def calculate_final_scores(self):
        for player in self.players:
            player.points += player.points_from_destinations()

    def current_player(self):
        return self.players[self.current_player_index]
