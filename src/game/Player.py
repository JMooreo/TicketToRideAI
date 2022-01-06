from typing import List

import numpy as np

from src.game.CardList import CardList


class Player:
    def __init__(self):
        self.points = 0
        self.trains = 45
        self.uncompleted_destinations = {}
        self.completed_destinations = {}
        self.routes = {}
        self.hand = CardList()
        self.turn_history = []
        self.long_term_turn_history = []

    def __str__(self):
        return f"Points: {self.points}\n" + \
               f"Points from routes: {self.points_from_routes()}\n" + \
               f"Points from destinations: {self.points_from_destinations()}\n" + \
               f"Trains Left: {self.trains}\n" + \
               f"Uncompleted Destinations: {self.uncompleted_destinations}\n" + \
               f"Completed Destinations: {self.completed_destinations}\n" + \
               f"Routes: {self.routes}\n" + \
               f"Hand: {self.hand}\n" + \
               f"Turn History {self.turn_history}\n"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Player) and \
               self.points == other.points and \
               self.trains == other.trains and \
               all([d in other.uncompleted_destinations for d in self.uncompleted_destinations]) and \
               all([d in other.completed_destinations for d in self.completed_destinations]) and \
               all([r in other.routes for r in self.routes]) and \
               self.hand == other.hand and \
               self.turn_history == other.turn_history

    def points_from_routes(self):
        return sum((route.points for route in self.routes.values()))

    def points_from_destinations(self):
        completed = sum(d.points for d in self.completed_destinations.values())
        uncompleted = sum(d.points for d in self.uncompleted_destinations.values())
        return completed - uncompleted

    def update_long_term_turn_history(self):
        action_ids = [action.id for action in self.turn_history]
        if any([id_ < 0 for id_ in action_ids]) < 0:
            raise ValueError

        encoded_turn_history = np.array([1 if id_ in action_ids else 0 for id_ in range(141)], dtype=np.float32)  # 141 is action space size, covered in test

        self.long_term_turn_history.append(encoded_turn_history)
        self.long_term_turn_history = self.long_term_turn_history[-3:]

    def get_last_turn(self):
        if len(self.long_term_turn_history) > 0:
            return self.long_term_turn_history[-1]

        return np.zeros(141, dtype=np.float32)  # 141 is action space length, covered by test
