from abc import abstractmethod
from typing import Dict, List, Tuple

from game.Destination import Destination
from game.Route import Route


class PathFinder:
    def __init__(self, available_routes: Dict[int, Route], destination: Destination, routes: Dict[int, Route]):
        self.available_routes = available_routes
        self.destination = destination
        self.routes = routes
        self.results: List[Tuple[List[Route], int]] = []

    @abstractmethod
    def search(self) -> None:
        pass

    @abstractmethod
    def get_optimal(self) -> Tuple[List[Route], int]:
        pass
