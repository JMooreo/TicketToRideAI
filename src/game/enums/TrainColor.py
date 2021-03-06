from fastenum import Enum


class TrainColor(Enum):
    def __str__(self):
        return self.name

    WHITE = 0
    BLUE = 1
    GREEN = 2
    ORANGE = 3
    PINK = 4
    RED = 5
    YELLOW = 6
    BLACK = 7
    WILD = 8
