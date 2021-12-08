from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class ClaimRouteAction(Action):

    def __init__(self, game: Game, route_id: int):
        super().__init__(game)
        self.route_id = route_id
        self.route = game.map.routes.get(route_id)
        self.player = game.players[self.game.current_player_index]

    def __str__(self):
        return f"claim_{str(self.route)} ({self.route.points} points)"

    def __eq__(self, other):
        return isinstance(other, ClaimRouteAction) and \
               self.game == other.game

    def is_valid(self):
        return 0 <= self.route_id < len(self.game.map.routes) and \
               self.game.turn_state == TurnState.INIT and \
               self.game.state in [GameState.PLAYING, GameState.LAST_ROUND] and \
               self.player.trains >= self.route.cost.amount and \
               self.route.cost.best_payment_option(self.player.hand) is not None and \
               self.route_id in self.game.unclaimed_routes and \
               self.route.adjacent_route_id not in self.player.routes

    def execute(self):
        super().execute()
        # TODO: let the AI choose how it wants to pay for a route
        #       instead of choosing the first valid option
        payment = self.route.cost.best_payment_option(self.player.hand)

        self.game.unclaimed_routes.pop(self.route_id)
        self.game.turn_state = TurnState.FINISHED
        self.game.deck += payment

        self.player.routes[self.route_id] = self.route
        self.player.points += self.route.points
        self.player.hand -= payment
        self.player.trains -= self.route.cost.amount

        keys_to_pop = []
        for key, destination in self.player.uncompleted_destinations.items():
            if destination.path_from(self.player.routes.values()) is not None:
                keys_to_pop.append(key)
                self.player.completed_destinations[key] = destination

        for key in keys_to_pop:
            self.player.uncompleted_destinations.pop(key)
