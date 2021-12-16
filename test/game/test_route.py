import unittest

from src.game.CardList import TrainColor
from src.game.enums.City import City
from src.game.Route import Route
from src.game.RouteCost import RouteCost


class RouteTest(unittest.TestCase):
    def test_init_route(self):
        route = Route([City.KANSAS_CITY, City.OKLAHOMA_CITY], RouteCost(TrainColor.BLACK, 6))

        self.assertEqual(City.KANSAS_CITY, route.cities[0])
        self.assertEqual(City.OKLAHOMA_CITY, route.cities[1])
        self.assertEqual(TrainColor.BLACK, route.cost.color)
        self.assertEqual(6, route.cost.amount)

    def test_init_route_with_adjacent_route(self):
        route = Route([City.KANSAS_CITY, City.OKLAHOMA_CITY], RouteCost(TrainColor.BLACK, 6), 56)

        self.assertEqual(56, route.adjacent_route_id)

    def test_points(self):
        lengths = [1, 2, 3, 4, 5, 6]
        expected_points = [1, 2, 4, 7, 10, 15]

        for idx, length in enumerate(lengths):
            expected = expected_points[idx]
            actual = Route([], RouteCost(TrainColor.BLACK, length)).points

            self.assertEqual(expected, actual)

    def test_length_below_minimum(self):
        with self.assertRaises(IndexError):
            cost = RouteCost(TrainColor.BLACK, 6)
            cost.amount = 0
            route = Route([], cost)

    def test_length_above_maximum(self):
        with self.assertRaises(IndexError):
            cost = RouteCost(TrainColor.BLACK, 6)
            cost.amount = 7
            route = Route([], cost)

    def test_has_city(self):
        route = Route([City.BOSTON, City.OKLAHOMA_CITY], RouteCost(TrainColor.BLACK, 1))

        self.assertTrue(route.has_city(City.BOSTON))
        self.assertTrue(route.has_city(City.OKLAHOMA_CITY))
        self.assertFalse(route.has_city(City.ATLANTA))

    def test_to_string(self):
        route = Route([City.BOSTON, City.OKLAHOMA_CITY], RouteCost(TrainColor.BLACK, 1))
        route2 = Route([City.CHARLESTON, City.DALLAS], RouteCost(TrainColor.GREEN, 6))

        self.assertEqual("BOSTON_to_OKLAHOMA_CITY (1 points)", str(route))
        self.assertEqual("CHARLESTON_to_DALLAS (15 points)", str(route2))

    def test_to_string_in_a_list(self):
        route = Route([City.BOSTON, City.OKLAHOMA_CITY], RouteCost(TrainColor.BLACK, 1))
        route2 = Route([City.CHARLESTON, City.DALLAS], RouteCost(TrainColor.GREEN, 6))
        routes = [route, route2]
        expected = f"[{str(route)}, {str(route2)}]"

        self.assertEqual(expected, str(routes))
