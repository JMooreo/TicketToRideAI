from typing import Dict, List

import numpy as np

from Environments.TTREnv import TTREnv
from actors.Agent import Agent
from algorithms.LeastCarsPathFinder import LeastCarsPathFinder
from game.CardList import CardList
from game.Route import Route


class NaiveAgent(Agent):
    def __init__(self):
        super().__init__()
        self.found_paths: Dict[int, List[Route]] = {}  # By destination id
        self.desired_routes = []
        self.desired_hand = CardList()

    # Can always assume that when an agent is acting, it is their turn.
    def act(self, env: TTREnv) -> int:
        desired_actions = np.zeros(len(env.action_space))
        current_player = env.tree.game.current_player()

        print(f"\n{self} wants to complete destinations: {current_player.uncompleted_destinations}")
        # Compute the routes needed
        desired_routes = set()

        for destination_id, destination in current_player.uncompleted_destinations.items():
            path_finder = LeastCarsPathFinder(env.tree.game.unclaimed_routes, destination, current_player.routes)
            path_finder.search(limit=current_player.trains)
            path, cost = path_finder.get_optimal()

            if path:
                for route in path:
                    desired_routes.add(route)

            print(f"\nFound path for destination {destination}. Cost: {cost}")
            print("Desired Routes:", desired_routes)

        # print(f"\n{self} wants to claim: {destination}")

        # Draw cards or complete routes as soon as it can. Prioritize claiming routes

        return env.action_space.sample()
