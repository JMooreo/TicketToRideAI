import unittest

from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player


class PlayerTest(unittest.TestCase):

    def setUp(self):
        self.game = Game([Player(), Player()], USMap())
        self.player = Player()
        self.destinations = self.game.map.destinations
        self.routes = self.game.map.routes

    def test_init(self):
        self.assertEqual(CardList(), self.player.hand)
        self.assertEqual({}, self.player.uncompleted_destinations)
        self.assertEqual({}, self.player.routes)
        self.assertEqual([], self.player.turn_history)

    def test_final_score_one_uncompleted_destination(self):
        self.player.uncompleted_destinations = {4: self.destinations.get(4)}
        expected_points = self.destinations.get(4).points * -1

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_two_uncompleted_destinations(self):
        self.player.uncompleted_destinations = {4: self.destinations.get(4), 5: self.destinations.get(5)}
        expected_points = (self.destinations.get(4).points + self.destinations.get(5).points) * -1

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_one_completed_destination(self):
        self.player.completed_destinations = {6: self.destinations.get(6)}
        self.player.routes = {32: self.routes.get(32), 33: self.routes.get(33)}
        expected_points = self.destinations.get(6).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_multiple_completed_destinations(self):
        self.player.completed_destinations = {i: self.destinations.get(i) for i in [6, 17]}
        self.player.routes = {i: self.routes.get(i) for i in [32, 33, 64, 17, 15]}

        expected_points = self.destinations.get(6).points + self.destinations.get(17).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_one_completed_and_one_failed_destination(self):
        self.player.completed_destinations = {6: self.destinations.get(6)}
        self.player.uncompleted_destinations = {17: self.destinations.get(17)}
        self.player.routes = {i: self.routes.get(i) for i in [32, 33]}

        expected_points = self.destinations.get(6).points - self.destinations.get(17).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_final_score_multiple_completed_destinations_with_starting_points(self):
        self.player.points = 20
        self.player.completed_destinations = {i: self.destinations.get(i) for i in [6, 17]}
        self.player.routes = {i: self.routes.get(i) for i in [32, 33, 64, 17, 15]}

        expected_points = self.destinations.get(6).points + self.destinations.get(17).points

        self.assertEqual(expected_points, self.player.points_from_destinations())

    def test_points_from_one_route(self):
        self.player.routes = {i: self.routes.get(i) for i in [14]}

        expected_points = self.routes.get(14).points

        self.assertEqual(expected_points, self.player.points_from_routes())

    def test_points_from_multiple_route(self):
        self.player.routes = {i: self.routes.get(i) for i in [14, 17]}

        expected_points = self.routes.get(14).points + self.routes.get(17).points

        self.assertEqual(expected_points, self.player.points_from_routes())
