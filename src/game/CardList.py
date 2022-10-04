import random

from src.game.enums.TrainColor import TrainColor
from typing import List


class CardList:
    def __init__(self, *colors: [(TrainColor, int)]):
        # Assumes no duplicates when creating a card list
        self.cards = {color: amount for color, amount in colors if amount > 0}

    @staticmethod
    def from_numbers(numbers: List[int]):
        cl = CardList()

        if len(numbers) > len(TrainColor):
            raise IndexError

        for index, number in enumerate(numbers):
            if number < 0:
                raise ValueError

            if number > 0:
                cl.cards[TrainColor(index)] = number

        return cl

    def __getitem__(self, color: TrainColor):
        return self.cards.get(color, 0)

    def __setitem__(self, color: TrainColor, amount: int):
        self.cards[color] = amount

    def __len__(self):
        return sum([val for _id, val in self.cards.items()])

    def __str__(self):
        return "_".join([f"{amount}{key.name}" for key, amount in self.cards.items() if amount > 0])

    def __add__(self, other):
        if not isinstance(other, CardList):
            other = other.as_cardlist()

        cards = {}

        for color in TrainColor:
            value = self.cards.get(color, 0) + other.cards.get(color, 0)
            if value > 0:
                cards[color] = value

        new_list = CardList()
        new_list.cards = cards
        return new_list

    def __sub__(self, other):
        cards = {}

        for color in TrainColor:
            value = self.cards.get(color, 0) - other.cards.get(color, 0)
            if value < 0:
                raise ValueError

            if value > 0:
                cards[color] = value

        new_list = CardList()
        new_list.cards = cards
        return new_list

    def subtract_clamp_at_zero(self, other):
        cards = {}

        for color in TrainColor:
            value = self.cards.get(color, 0) - other.cards.get(color, 0)
            cards[color] = max(value, 0)

        new_list = CardList()
        new_list.cards = cards
        return new_list

    def __eq__(self, other):
        return self.cards == other.cards

    def __repr__(self):
        return str(self)

    def __copy__(self):
        new_list = CardList()
        new_list.cards = self.cards
        return new_list

    def number_of_cards(self):
        return sum(self.cards.values())

    def has(self, other):
        for color, amount in other.cards.items():
            if self.cards.get(color, 0) < amount:
                return False

        return True

    def get_random(self, amount: int):
        card_list = CardList()
        for i in range(amount):
            card_list += self.draw_train_card()

        return card_list

    def draw_train_card(self):
        if self.number_of_cards() > 0:
            color = random.choice(list(self.cards.keys()))
            assert self.cards[color] > 0
            if self.cards[color] == 1:
                self.cards.pop(color)
            else:
                self.cards[color] -= 1
            return CardList((color, 1))

        return CardList()
