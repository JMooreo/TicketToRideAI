from typing import List

from src.game.enums.TrainColor import TrainColor
from src.game.enums.City import City
from src.game.Route import Route
from src.game.RouteCost import RouteCost


class Destination:
    def __init__(self, cities: List[City], points: int):
        self.points = points
        self.cities = cities

    def __str__(self):
        return str(self.cities[0]) + "_to_" + str(self.cities[1]) + f" ({self.points} points)"

    def __repr__(self):
        return str(self)

    def path_from(self, routes):
        return self.search(path=[Route([self.cities[0], self.cities[0]], RouteCost(TrainColor.WILD, 1))],
                           visited=[],
                           goal=self.cities[1],
                           routes_to_check=routes)

    def search(self, path: List[Route], visited: List[City], goal: City, routes_to_check: List[Route]):
        # Find which city on the path hasn't been visited yet
        next_unvisited = next((city for city in path[-1].cities if city not in visited), None)

        matched_routes = [r for r in routes_to_check
                          if r.has_city(next_unvisited) and
                          r.cities[0] not in visited and
                          r.cities[1] not in visited]

        # If there are no matches, this path is dead
        if len(matched_routes) == 0:
            return None

        # Update the visited list with the last city we checked
        visited.append(next_unvisited)
        results = []

        for r in matched_routes:
            # Base Case, we're done searching
            if r.has_city(goal):
                path.append(r)
                path.pop(0)
                return path

            # Recursive case, find results from the current known branches
            path.append(r)
            results.append(self.search(path, visited, goal, routes_to_check))

        # From those results, see if there is a valid path
        for r in results:
            if r is not None:
                return r
