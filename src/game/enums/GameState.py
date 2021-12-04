from enum import Enum


class GameState(Enum):
    FIRST_TURN = 0
    PLAYING = 1
    LAST_TURN = 2
    GAME_OVER = 3
