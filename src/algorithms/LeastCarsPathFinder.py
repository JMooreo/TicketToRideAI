from typing import List, Tuple, Dict

import numpy as np

from algorithms.PathFinder import PathFinder
from game.Destination import Destination
from game.Route import Route
from game.enums.City import City


class LeastCarsPathFinder(PathFinder):

    def __init__(self, available_routes: Dict[int, Route], destination: Destination, routes: Dict[int, Route]):
        super().__init__(available_routes, destination, routes)
        self.branch_queue: List[Tuple[City, set, int, list[Route]]] = []

    def get_optimal(self) -> Tuple[List[Route], int]:
        true_cost_results = []

        # Account for the routes that we already own.
        for path, cost in self.results:
            actual_cost = cost
            for route in path:
                if route in self.routes.values():
                    actual_cost -= route.cost.amount

            true_cost_results.append((path, actual_cost))

        if len(true_cost_results) == 0:
            return [], np.inf

        return min(true_cost_results, key=lambda result: result[1])

    def add_branches_to_the_branch_queue(self, current, visited, cost, path, limit):
        for route in self.available_routes.values():
            city1, city2 = route.cities

            if (
                    not route.has_city(current)
                    or (city1 in visited and city2 in visited)  # Dead end
                    or cost > max([cost for path, cost in self.results], default=limit)  # Too expensive
                    or route.adjacent_route_id in self.routes  # Game rules
            ):
                continue

            next_city = city1 if city1 not in visited else city2
            _next = (next_city, {*visited, next_city}, cost + route.cost.amount, [*path, route])

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
            current, visited, cost, path = sorted(self.branch_queue, key=lambda branch: branch[2])[0]  # By cost
            self.branch_queue = self.branch_queue[1:]

            if current == goal:
                self.results.append((path, cost))
            else:
                max_found = max([cost for path, cost in self.results], default=limit)
                if cost < max_found:
                    self.add_branches_to_the_branch_queue(current, visited, cost, path, min(max_found, limit))
