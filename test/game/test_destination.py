import unittest

from src.game.CardList import TrainColor
from src.game.enums.City import City
from src.game.Destination import Destination
from src.game.Route import Route
from src.game.RouteCost import RouteCost


class DestinationTest(unittest.TestCase):
    def test_init(self):
        destination = Destination([City.BOSTON, City.DULUTH], 21)

        self.assertEqual([City.BOSTON, City.DULUTH], destination.cities)
        self.assertEqual(21, destination.points)

    def test_adjacent_routes_complete_a_destination(self):
        destination = Destination([City.DENVER, City.EL_PASO], 4)
        routes = [
            Route([City.SANTA_FE, City.DENVER], RouteCost(TrainColor.WILD, 2)),
            Route([City.SANTA_FE, City.EL_PASO], RouteCost(TrainColor.WILD, 2))
        ]

        self.assertEqual(str(routes), str(destination.path_from(routes)))

    def test_non_adjacent_routes_do_not_complete_a_destination(self):
        destination = Destination([City.DENVER, City.EL_PASO], 4)
        routes = [
            Route([City.OKLAHOMA_CITY, City.DENVER], RouteCost(TrainColor.WILD, 2)),
            Route([City.SANTA_FE, City.EL_PASO], RouteCost(TrainColor.WILD, 2))
        ]

        self.assertIsNone(destination.path_from(routes))

    def test_three_step_path_completes_a_destination(self):
        destination = Destination([City.DENVER, City.EL_PASO], 4)
        routes = [
            Route([City.OKLAHOMA_CITY, City.DENVER], RouteCost(TrainColor.WILD, 2)),
            Route([City.OKLAHOMA_CITY, City.SANTA_FE], RouteCost(TrainColor.RED, 5)),
            Route([City.SANTA_FE, City.EL_PASO], RouteCost(TrainColor.WILD, 2))
        ]

        self.assertIsNotNone(destination.path_from(routes))

    def test_very_long_path_completes_multiple_destinations(self):
        destination1 = Destination([City.BOSTON, City.NEW_YORK], 4)
        destination2 = Destination([City.DULUTH, City.NEW_YORK], 21)
        routes = [
            Route([City.BOSTON, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2)),
            Route([City.OKLAHOMA_CITY, City.DULUTH], RouteCost(TrainColor.WILD, 2)),
            Route([City.DULUTH, City.CALGARY], RouteCost(TrainColor.WILD, 2)),
            Route([City.CALGARY, City.ATLANTA], RouteCost(TrainColor.WILD, 2)),
            Route([City.ATLANTA, City.NEW_YORK], RouteCost(TrainColor.WILD, 2))
        ]

        self.assertIsNotNone(destination1.path_from(routes))
        self.assertIsNotNone(destination2.path_from(routes))

    def test_destination_not_completed_if_staring_city_not_in_routes(self):
        destination = Destination([City.BOSTON, City.NEW_YORK], 4)
        routes = [
            Route([City.NEW_YORK, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2)),
            Route([City.ATLANTA, City.NEW_YORK], RouteCost(TrainColor.WILD, 2)),
        ]

        self.assertIsNone(destination.path_from(routes))

    def test_destination_not_completed_if_goal_city_not_in_routes(self):
        destination = Destination([City.BOSTON, City.NEW_YORK], 4)
        routes = [
            Route([City.BOSTON, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2)),
            Route([City.ATLANTA, City.BOSTON], RouteCost(TrainColor.WILD, 2)),
        ]

        self.assertIsNone(destination.path_from(routes))

    def test_destination_not_completed_by_weird_routes_that_point_at_themselves(self):
        destination = Destination([City.BOSTON, City.NEW_YORK], 4)
        routes = [
            Route([City.BOSTON, City.BOSTON], RouteCost(TrainColor.WILD, 2)),
            Route([City.NEW_YORK, City.NEW_YORK], RouteCost(TrainColor.WILD, 2)),
        ]

        self.assertIsNone(destination.path_from(routes))
