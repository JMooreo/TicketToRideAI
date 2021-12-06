from enum import Enum


class GameState(Enum):
    FIRST_ROUND = 0
    PLAYING = 1
    LAST_ROUND = 2
    GAME_OVER = 3
