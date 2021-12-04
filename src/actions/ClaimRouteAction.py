from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState


class ClaimRouteAction(Action):

    def __init__(self, game: Game, route_id: int):
        if not isinstance(route_id, int):
            raise ValueError

        if not 0 <= route_id < len(game.map.routes):
            raise IndexError

        super().__init__(game)
        self.route_id = route_id
        self.route = game.map.routes.get(route_id)
        self.player = game.players[self.game.current_player_index]

    def __str__(self):
        return f"claim_{self.route_id}"

    def is_valid(self):
        return self.game.turn_state == TurnState.INIT and \
               self.game.state in [GameState.PLAYING, GameState.LAST_TURN] and \
               self.route.cost.best_payment_option(self.player.hand) is not None and \
               self.route_id in self.game.unclaimed_routes and \
               self.route.adjacent_route_id not in self.player.owned_routes

    def execute(self):
        self.game.unclaimed_routes.pop(self.route_id)
        self.game.turn_state = TurnState.FINISHED
        self.player.owned_routes.append(self.route_id)
        self.player.points += self.route.points
        self.player.hand -= self.route.cost.best_payment_option(self.player.hand)
        self.player.trains -= self.route.cost.amount
