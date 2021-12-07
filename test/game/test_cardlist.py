import copy
import unittest
from src.game.CardList import CardList, TrainColor


class CardListTest(unittest.TestCase):

    def setUp(self):
        self.card_list = CardList.from_numbers([i for i in range(len(TrainColor))])

    def test_initialization_empty(self):
        self.assertEqual({}, CardList().cards)

    def test_initialization_one_card(self):
        card_list = CardList((TrainColor.BLACK, 1))

        self.assertEqual(1, card_list[TrainColor.BLACK])

    def test_initialization_multiple_cards(self):
        card_list = CardList((TrainColor.BLACK, 2))

        self.assertEqual(2, card_list[TrainColor.BLACK])

    def test_initialization_multiple_colors(self):
        card_list = CardList((TrainColor.BLACK, 1), (TrainColor.YELLOW, 1))

        self.assertEqual(1, card_list[TrainColor.BLACK])
        self.assertEqual(1, card_list[TrainColor.YELLOW])

    def test_initialization_multiple_colors_multiple_cards(self):
        card_list = CardList((TrainColor.BLACK, 2), (TrainColor.YELLOW, 3), (TrainColor.GREEN, 5))

        self.assertEqual(2, card_list[TrainColor.BLACK])
        self.assertEqual(3, card_list[TrainColor.YELLOW])
        self.assertEqual(5, card_list[TrainColor.GREEN])

    def test_from_numbers_valid(self):
        card_list = CardList.from_numbers([0, 1, 2, 3, 4, 5, 6, 7, 8])

        for idx, color in enumerate(TrainColor):
            self.assertEqual(idx, card_list[color])

    def test_from_numbers_too_long(self):
        with self.assertRaises(IndexError):
            CardList.from_numbers([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_from_numbers_negative(self):
        with self.assertRaises(ValueError):
            CardList.from_numbers([-1])

    def test_length_one(self):
        self.assertEqual(1, len(CardList.from_numbers([1])))

    def test_length_multiple(self):
        self.assertEqual(2, len(CardList.from_numbers([1, 1])))

    def test_string(self):
        expected = str({color: i + 1 for i, color in enumerate(list(TrainColor)[1:])})
        self.assertEqual(expected, str(self.card_list))

    def test_string_with_values(self):
        card_list = CardList.from_numbers([1, 2, 3])
        expected = str({color: i + 1 for i, color in enumerate(list(TrainColor)[:3])})

        self.assertEqual(expected, str(card_list))

    def test_add_another_cardlist(self):
        card_list = CardList.from_numbers([0, 1, 2])
        card_list2 = CardList.from_numbers([0, 0, 0, 3, 4, 5])

        print(card_list)
        print(card_list2)

        expected = {TrainColor.BLUE: 1, TrainColor.GREEN: 2, TrainColor.ORANGE: 3, TrainColor.PINK: 4,
                    TrainColor.RED: 5}

        actual = (card_list + card_list2).cards

        self.assertEqual(expected, actual)

    def test_add_None(self):
        with self.assertRaises(AttributeError):
            card_list = CardList.from_numbers([0, 1, 2])
            actual = card_list + None

    def test_add_a_list_not_a_card_list(self):
        with self.assertRaises(AttributeError):
            card_list = CardList.from_numbers([0, 1, 2])
            actual = card_list + [0, 1, 2, 3, 4]

    def test_subtract(self):
        card_list = CardList.from_numbers([0, 1, 2])
        card_list2 = CardList.from_numbers([0, 1, 1])

        expected = CardList.from_numbers([0, 0, 1, 0, 0, 0, 0, 0, 0])
        actual = card_list - card_list2

        self.assertEqual(expected, actual)

    def test_subtract_cant_go_negative(self):
        card_list = CardList.from_numbers([0, 0, 0])
        card_list2 = CardList.from_numbers([1, 1, 1])

        with self.assertRaises(ValueError):
            result = card_list - card_list2

    def test_subtract_None(self):
        with self.assertRaises(AttributeError):
            card_list = CardList.from_numbers([0, 1, 2])
            actual = card_list - None

    def test_subtract_a_list_not_a_card_list(self):
        with self.assertRaises(AttributeError):
            card_list = CardList.from_numbers([0, 1, 2])
            actual = card_list - [0, 1, 2, 3, 4]

    def test_repr(self):
        card_list = CardList.from_numbers([0, 1])
        card_list2 = CardList.from_numbers([0])

        expected = f"[{str(card_list)}, {str(card_list2)}]"

        self.assertEqual(expected, str([card_list, card_list2]))

    def test_copy(self):
        list_copy = copy.copy(self.card_list)

        self.assertIsNot(self.card_list, list_copy)
        self.assertEqual(self.card_list, list_copy)

    def test_train_color_below_minimum(self):
        list_copy = copy.copy(self.card_list)

        self.assertIsNot(self.card_list, list_copy)
        self.assertEqual(self.card_list, list_copy)

    def test_has_none(self):
        card_list = CardList((TrainColor.BLACK, 3), (TrainColor.GREEN, 4), (TrainColor.WILD, 2))

        self.assertFalse(card_list.has(None))

    def test_has_one_card(self):
        card_list = CardList((TrainColor.BLACK, 3), (TrainColor.GREEN, 4), (TrainColor.WILD, 2))

        self.assertTrue(card_list.has(CardList((TrainColor.BLACK, 1))))

    def test_does_not_have_one_card(self):
        card_list = CardList((TrainColor.BLACK, 3), (TrainColor.GREEN, 4), (TrainColor.WILD, 2))

        self.assertFalse(card_list.has(CardList((TrainColor.ORANGE, 1))))

    def test_get_random_from_empty(self):
        card_list = CardList()

        self.assertEqual(CardList(), card_list.get_random(1))

    def test_get_random_from_more_than_what_exists_in_the_card_list(self):
        card_list = CardList((TrainColor.YELLOW, 3))

        self.assertEqual(CardList((TrainColor.YELLOW, 3)), card_list.get_random(4))

    def test_init_with_the_same_color_more_than_once(self):
        card_list = CardList((TrainColor.YELLOW, 3), (TrainColor.YELLOW, 3))

        self.assertEqual(CardList((TrainColor.YELLOW, 6)), card_list)

    def test_number_of_cards_empty_list(self):
        card_list = CardList()

        self.assertEqual(0, card_list.number_of_cards())

    def test_multiple_cards_of_one_color(self):
        card_list = CardList((TrainColor.WILD, 3))

        self.assertEqual(3, card_list.number_of_cards())

    def test_multiple_different_colors(self):
        card_list = CardList((TrainColor.WILD, 3), (TrainColor.BLACK, 3))

        self.assertEqual(6, card_list.number_of_cards())
