from src.game.Game import Game


class InformationSet:
    @staticmethod
    def from_game(game: Game, player_index: int):
        return str(sorted({key: dest for key, dest in game.players[player_index].uncompleted_destinations.items()}))
