import unittest

from src.game.CardList import TrainColor, CardList
from src.game.RouteCost import RouteCost


class RouteCostTest(unittest.TestCase):

    def test_length_below_minimum(self):
        with self.assertRaises(IndexError):
            RouteCost(TrainColor.BLACK, 0)

    def test_length_above_maximum(self):
        with self.assertRaises(IndexError):
            RouteCost(TrainColor.BLACK, 7)

    def test_train_color_above_maximum(self):
        with self.assertRaises(ValueError):
            RouteCost(TrainColor(9), 7)

    def test_train_color_below_minimum(self):
        with self.assertRaises(ValueError):
            RouteCost(TrainColor(-1), 7)

    def test_train_color_minimum_index(self):
        cost = RouteCost(TrainColor(0), 3)

        self.assertIsNotNone(cost)

    def test_train_color_maximum_index(self):
        cost = RouteCost(TrainColor(len(TrainColor)-1), 3)

        self.assertIsNotNone(cost)

    def test_route_cost_init(self):
        route_cost = RouteCost(TrainColor.BLACK, 6)

        self.assertEqual(TrainColor.BLACK, route_cost.color)
        self.assertEqual(6, route_cost.amount)

    def test_init_with_bad_train_color_below_minimum(self):
        with self.assertRaises(ValueError):
            RouteCost(-1, 6)

    def test_init_with_bad_train_color_above_maximum(self):
        with self.assertRaises(ValueError):
            RouteCost(len(TrainColor), 6)

    def test_pay_with_exact_color_and_amount(self):
        route_cost = RouteCost(TrainColor.BLUE, 4)

        expected = CardList((TrainColor.BLUE, 4))
        actual = route_cost.best_payment_option(CardList((TrainColor.BLUE, 4)))

        self.assertEqual(expected, actual)

    def test_pay_with_wrong_color_right_amount(self):
        route_cost = RouteCost(TrainColor.BLUE, 4)

        actual = route_cost.best_payment_option(CardList((TrainColor.GREEN, 4)))

        self.assertIsNone(actual)

    def test_pay_with_wilds(self):
        route_cost = RouteCost(TrainColor.BLUE, 2)

        actual = route_cost.best_payment_option(CardList((TrainColor.WILD, 2)))
        expected = CardList((TrainColor.WILD, 2))

        self.assertEqual(expected, actual)

    def test_can_pay_with_leftover_cards(self):
        route_cost = RouteCost(TrainColor.BLUE, 1)

        expected = CardList((TrainColor.WILD, 1))
        actual = route_cost.best_payment_option(CardList((TrainColor.WILD, 2)))

        self.assertEqual(expected, actual)

    def test_prefer_colors_over_wilds(self):
        route_cost = RouteCost(TrainColor.BLUE, 5)
        hand = CardList((TrainColor.BLUE, 4), (TrainColor.WILD, 4))

        expected = CardList((TrainColor.BLUE, 4), (TrainColor.WILD, 1))
        actual = route_cost.best_payment_option(hand)

        self.assertEqual(expected, actual)

    def test_pay_for_wilds_with_other_colors(self):
        route_cost = RouteCost(TrainColor.WILD, 1)
        hand = CardList.from_numbers([4, 4, 4, 4, 4, 4, 4, 0])

        expected = CardList.from_numbers([1, 0, 0, 0, 0, 0, 0, 0])
        actual = route_cost.best_payment_option(hand)

        self.assertEqual(expected, actual)

    def test_pay_for_wild_with_multiple_colors(self):
        route_cost = RouteCost(TrainColor.WILD, 2)
        hand = CardList.from_numbers([1, 1, 0, 0, 0, 0, 0, 0])

        actual = route_cost.best_payment_option(hand)

        self.assertIsNone(actual)


