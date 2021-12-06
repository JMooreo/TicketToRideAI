from src.game.CardList import CardList


class Player:
    def __init__(self):
        self.points = 0
        self.trains = 45
        self.destinations = {}
        self.routes = {}
        self.hand = CardList()
        self.turn_history = []

    def __str__(self):
        return f"Points: {self.points}\n" + \
                f"Trains Left: {self.trains}\n" + \
                f"Destinations: {self.destinations}\n" + \
                f"Routes: {self.routes}\n" + \
                f"Hand: {self.hand}\n" + \
                f"Turn History {self.turn_history}\n"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Player) and \
                self.points == other.points and \
                self.trains == other.trains and \
                all([d in other.destinations for d in self.destinations]) and \
                all([r in other.routes for r in self.routes]) and \
                self.hand == other.hand and \
                self.turn_history == other.turn_history

    def points_from_routes(self):
        return sum((route.points for route in self.routes.values()))

    def points_from_destinations(self):
        return sum((destination.points if destination.path_from(self.routes.values()) is not None
                   else -destination.points for destination in self.destinations.values()))
