from typing import Generator

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

    def as_cardlist(self):
        return CardList((self.color, self.amount))

    def __add__(self, other):
        if isinstance(other, RouteCost):
            return self.as_cardlist() + other.as_cardlist()

        if isinstance(other, CardList):
            return self.as_cardlist() + other

        raise TypeError(f"Could not add {type(other)} to type RouteCost.")

    def __str__(self):
        return str(CardList((self.color, self.amount)))

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
