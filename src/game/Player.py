from src.game.CardList import CardList


class Player:
    def __init__(self):
        self.points = 0
        self.trains = 45
        self.owned_destinations = []
        self.owned_routes = []
        self.hand = CardList()
