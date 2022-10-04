from typing import List, Tuple, Dict

import numpy as np

from algorithms.PathFinder import PathFinder
from game.Destination import Destination
from game.Route import Route
from game.enums.City import City


class LeastCarsPathFinder(PathFinder):

    def __init__(self, unclaimed_routes: Dict[int, Route], destination: Destination, routes: Dict[int, Route]):
        super().__init__(unclaimed_routes, destination, routes)
        self.branch_queue: List[Tuple[City, set, int, list[Route]]] = []

    def get_optimal(self) -> Tuple[List[Route], int]:
        if len(self.results) == 0:
            return [], np.inf

        return min(self.results, key=lambda result: result[1])

    def add_branches_to_the_branch_queue(self, current, visited, cost, path, limit):
        for route in self.available_routes.values():
            city1, city2 = route.cities

            if (
                    not route.has_city(current)
                    or (city1 in visited and city2 in visited)  # Dont go backward
                    or cost > max([cost for path, cost in self.results], default=limit)  # Already have cheaper options
                    or route.adjacent_route_id in self.owned_routes  # Game rules
            ):
                continue

            next_city = city1 if city1 not in visited else city2
            true_route_cost = route.cost.amount if route not in self.owned_routes.values() else 0
            _next = (next_city, {*visited, next_city}, cost + true_route_cost, [*path, route])

            if _next not in self.branch_queue:
                self.branch_queue.append(_next)

    def search(self, limit=45):
        path = []
        start = self.destination.cities[0]
        goal = self.destination.cities[1]
        current = start
        visited = {start}

        self.add_branches_to_the_branch_queue(current, visited, 0, path, limit)

        while len(self.branch_queue) > 0:
            self.branch_queue = sorted(self.branch_queue, key=lambda branch: branch[2])  # By cost
            current, visited, cost, path = self.branch_queue[0]
            self.branch_queue = self.branch_queue[1:]

            if current == goal:
                self.results.append((path, cost))

            max_found = max([cost for path, cost in self.results], default=limit)

            if cost < max_found:
                self.add_branches_to_the_branch_queue(current, visited, cost, path, min(max_found, limit))
