from enum import Enum


class TurnState(Enum):
    INIT = 0
    SELECTING_DESTINATIONS = 1
    DRAWING_CARDS = 2
    FINISHED = 3
