import numpy as np

from src.game.CardList import CardList


class Player:
    def __init__(self, game):
        self.game = game
        self.points = 0
        self.trains = 45
        self.uncompleted_destinations = {}
        self.completed_destinations = {}
        self.routes = {}
        self.hand = CardList()
        self.turn_history = []
        self.memory = []

    def __str__(self):
        return f"Points: {self.points}\n" + \
               f"Points from routes: {self.points_from_routes()}\n" + \
               f"Points from destinations: {self.points_from_destinations()}\n" + \
               f"Trains Left: {self.trains}\n" + \
               f"Uncompleted Destinations: {self.uncompleted_destinations}\n" + \
               f"Completed Destinations: {self.completed_destinations}\n" + \
               f"Routes: {self.routes}\n" + \
               f"Hand: {self.hand}\n" + \
               f"Current Turn History: {self.turn_history}\n" + \
               f"Memory: {self.human_readable_memory()}\n"

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

    def human_readable_memory(self):
        from training.ActionSpace import ActionSpace

        turns = []
        for one_hot_array in self.memory:
            action_ids = np.argwhere(one_hot_array == 1).squeeze()
            try:
                action_ids = list(action_ids)
                # Took multiple actions
            except Exception as e:
                # Only took one action
                action_ids = [action_ids]

            turns.append([str(ActionSpace(self.game).get_action_by_id(i)) for i in action_ids])

        return turns

    def points_from_routes(self):
        return sum((route.points for route in self.routes.values()))

    def points_from_destinations(self):
        completed = sum(d.points for d in self.completed_destinations.values())
        uncompleted = sum(d.points for d in self.uncompleted_destinations.values())
        return completed - uncompleted

    def update_memory(self, limit=0):
        action_ids = [action.id for action in self.turn_history]
        if any([id_ < 0 for id_ in action_ids]) < 0:
            raise ValueError

        encoded_turn_history = np.array([1 if id_ in action_ids else 0 for id_ in range(141)], dtype=np.float32)  # 141 is action space size, covered in test

        self.memory.append(encoded_turn_history)

        if limit > 0:
            self.memory = self.memory[-limit:]

    def get_last_turn(self):
        if len(self.memory) > 0:
            return self.memory[-1]

        return np.zeros(141, dtype=np.float32)  # 141 is action space length, covered by test
