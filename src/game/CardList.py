import random

from src.game.enums.TrainColor import TrainColor
from typing import List


class CardList:
    def __init__(self, *colors: [(TrainColor, int)]):
        self.list = [0 for _ in TrainColor]

        for color, amount in colors:
            self.list[TrainColor(color).value] += amount

    @staticmethod
    def from_numbers(numbers: List[int]):
        cl = CardList()
        cl.list = [0 for _ in TrainColor]

        if len(numbers) > len(TrainColor):
            raise IndexError

        for index, number in enumerate(numbers):
            if number < 0:
                raise ValueError

            cl.list[index] = number

        return cl

    def __getitem__(self, color: TrainColor):
        return self.list[color]

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def __add__(self, other):
        return CardList.from_numbers([x + y for x, y in zip(self.list, other.list)])

    def __sub__(self, other):
        return CardList.from_numbers([x - y for x, y in zip(self.list, other.list)])

    def __eq__(self, other):
        return self.list == other.list

    def __repr__(self):
        return str(self)

    def __copy__(self):
        return CardList() + self

    def number_of_cards(self):
        return sum(self.list)

    def has(self, other):
        if other is None:
            return False
        try:
            return all([x - y >= 0 for x, y in zip(self.list, other.list)])
        except ValueError:
            return False

    def get_random(self, amount: int):
        card_list = CardList()
        for i in range(amount):
            card_list += self.draw_train_card()

        return card_list

    def draw_train_card(self):
        if sum(self.list) > 0:
            color = random.choice(list(TrainColor))
            choice = CardList((color, 1))
            if self.list[color] > 0:
                self.list[color] -= 1
                return choice
            else:
                return self.draw_train_card()  # No cards of that color, try again.
        else:
            # No more cards to draw
            return CardList()
