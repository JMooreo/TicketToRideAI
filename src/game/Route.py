from typing import List
from src.game.enums.City import City
from src.game.RouteCost import RouteCost


class Route:
    def __init__(self, cities: List[City], cost: RouteCost, adjacent_route_id=None):
        if cost.amount <= 0:
            raise IndexError

        self.adjacent_route_id = adjacent_route_id
        self.points = [1, 2, 4, 7, 10, 15][cost.amount - 1]
        self.cost = cost
        self.cities = cities

    def has_city(self, city: City) -> bool:
        return city in self.cities

    def __str__(self):
        return str(self.cities[0]) + "_to_" + str(self.cities[1])

    def __repr__(self):
        return str(self)
