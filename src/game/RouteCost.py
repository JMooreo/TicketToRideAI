from typing import List, Generator

from src.game.CardList import TrainColor, CardList


class RouteCost:
    def __init__(self, color: TrainColor, amount: int):
        if not isinstance(color, TrainColor):
            raise ValueError

        if amount <= 0 or amount > 6:
            raise IndexError

        if color.value < 0 or color.value >= len(TrainColor):
            raise IndexError

        self.color = color
        self.amount = amount

    def __str__(self):
        return str(next(self.__options()))

    def best_payment_option(self, card_list: CardList):
        return next((option for option in self.__options() if card_list.has(option)), None)

    # Purposefully orders the options to preserve wild cards if possible. Perhaps there is a slight bias here
    # but the agent does not select how to pay for its routes on its own.
    # so this gives it a small helping hand.
    def __options(self) -> Generator[CardList, None, None]:
        if self.color != TrainColor.WILD:
            return (CardList((self.color, i), (TrainColor.WILD, self.amount - i)) for i in range(self.amount, -1, -1))
        else:
            return (CardList((c, self.amount)) for c in list(TrainColor))
