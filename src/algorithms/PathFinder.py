from abc import abstractmethod
from typing import Dict, List, Tuple

from game.Destination import Destination
from game.Route import Route


class PathFinder:
    def __init__(self, unclaimed_routes: Dict[int, Route], destination: Destination, owned_routes: Dict[int, Route]):
        self.available_routes = {**unclaimed_routes, **owned_routes}
        self.destination = destination
        self.owned_routes = owned_routes  # Subtract these from the cost to calculate the true cost.
        self.results: List[Tuple[List[Route], int]] = []

    @abstractmethod
    def search(self) -> None:
        pass

    @abstractmethod
    def get_optimal(self) -> Tuple[List[Route], int]:
        pass
