from typing import Dict, List

import numpy as np

from Environments.TTREnv import TTREnv
from actors.Agent import Agent
from algorithms.LeastCarsPathFinder import LeastCarsPathFinder
from game.CardList import CardList
from game.Route import Route
from game.enums.TrainColor import TrainColor
from training.ActionSpace import ActionSpace


class NaiveAgent(Agent):
    def __init__(self):
        super().__init__()
        self.desired_routes: Dict[int, List[Route]] = {}
        self.desired_hand = CardList()
        # Performance Bump - The number of routes that were unclaimed last time we searched for routes.
        self.previous_unclaimed_routes: Dict[int, Dict[int, Route]] = {}

    # Can always assume that when an agent is acting, it is their turn.
    def act(self, env: TTREnv) -> int:
        desired_actions = np.zeros(len(env.action_space))
        current_player = env.tree.game.current_player()

        # print(f"\n{self} wants to complete destinations: {current_player.uncompleted_destinations}")
        # Compute the routes needed
        desired_hand = CardList()
        for destination_id, destination in current_player.uncompleted_destinations.items():
            # Check if unclaimed routes have changed since last time
            if self.previous_unclaimed_routes.get(destination_id, None) != env.tree.game.unclaimed_routes:
                path_finder = LeastCarsPathFinder(env.tree.game.unclaimed_routes, destination, current_player.routes)
                path_finder.search(limit=current_player.trains)
                path, cost = path_finder.get_optimal()

                self.previous_unclaimed_routes[destination_id] = env.tree.game.unclaimed_routes

                if path:
                    for route in path:
                        if route not in current_player.routes.values():
                            self.desired_routes[destination_id] = path

                # print(f"\nFound path for destination {destination}. Cost: {cost}")

            desired_hand += sum([route.cost for route in self.desired_routes.get(destination_id, [])], CardList())

        # print("Desired Routes:", self.desired_routes)
        # print("Desired Hand:", desired_hand.subtract_clamp_at_zero(current_player.hand))

        action_space = ActionSpace(env.tree.game)

        # TODO: Fix these action ids and put them into some giant ENUM

        # Draw Destinations
        action = action_space.get_action_by_id(0)
        if action.is_valid() and self.desired_hand != CardList():
            return action.id

        # Finish Selecting Destinations
        action = action_space.get_action_by_id(1)
        if action.is_valid() and current_player.trains < 15:
            return action.id

        # Select Destinations
        offset = 3 + len(TrainColor) + len(env.tree.game.map.routes)
        sorted_ids_by_points = sorted([_id for _id in env.tree.game.available_destinations],
                                      key=lambda dest_id: env.tree.game.map.destinations[dest_id].points)
        for destination_id in sorted_ids_by_points:
            action = action_space.get_action_by_id(offset + destination_id)
            if action.is_valid():
                return action.id

        # Claim Routes
        offset = 3 + len(TrainColor)
        all_desired_routes = set([route for routes in self.desired_routes.values() for route in routes])
        for route in all_desired_routes:
            action = action_space.get_action_by_id(offset + route.id)
            if action.is_valid():
                return action.id

        # Draw Desired Colored Cards
        offset = 3
        for i, color in enumerate(TrainColor):
            if color.value != TrainColor.WILD and desired_hand.has(CardList((color.value, 1))):
                action = action_space.get_action_by_id(offset + i)
                if action.is_valid():
                    return action.id

        # Draw WILDS
        action = action_space.get_action_by_id(3 + len(TrainColor))
        if action.is_valid() and len(current_player.uncompleted_destinations) > 0:
            return action.id

        # Draw Random
        action = action_space.get_action_by_id(2)
        if action.is_valid() and len(current_player.uncompleted_destinations) > 0:
            return action.id

        return env.action_space.sample()
