import unittest

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.actions.DrawWildCardAction import DrawWildCardAction
from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState
from src.training.GameNode import OpponentNode, TrainingNode
from src.training.GameTree import GameTree


class OpponentNodeTest(unittest.TestCase):

    def setUp(self):
        self.players = [Player(), Player()]
        self.game = Game(self.players, USMap())
        self.tree = GameTree(self.game)

    def __do_first_turn(self):
        for _ in range(2):
            self.tree.next(DrawDestinationsAction(self.game))
            for _ in range(3):
                self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

    def __do_training_node_turn(self):
        self.tree.next(DrawDestinationsAction(self.game))
        for _ in range(3):
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

    def test_opponent_draw_destinations_after_first_turn(self):
        self.__do_first_turn()
        self.tree.next(ClaimRouteAction(self.game, 2))

        self.tree.next(DrawDestinationsAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, OpponentNode))
        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.game.turn_state)
        self.assertEqual(3, len(self.game.available_destinations))
        self.assertEqual(3, len(self.players[1].uncompleted_destinations))

    def test_opponent_draw_and_select_destinations_after_first_turn(self):
        self.__do_first_turn()
        self.tree.next(ClaimRouteAction(self.game, 2))

        self.tree.next(DrawDestinationsAction(self.game))
        for _ in range(3):
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.assertTrue(isinstance(self.tree.current_node, TrainingNode))
        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertEqual([], self.game.available_destinations)
        self.assertEqual(6, len(self.players[1].uncompleted_destinations))

    def test_opponent_draw_random_card_after_first_turn_done(self):
        self.__do_first_turn()
        self.tree.next(ClaimRouteAction(self.game, 2))

        self.tree.next(DrawRandomCardAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, OpponentNode))
        self.assertTrue(self.game.turn_state == TurnState.DRAWING_CARDS)
        self.assertEqual(3, self.game.players[0].hand.number_of_cards())
        self.assertEqual(5, self.game.players[1].hand.number_of_cards())

    def test_opponent_draw_wild_after_first_turn(self):
        self.__do_first_turn()
        self.tree.next(ClaimRouteAction(self.game, 2))
        self.game.visible_cards.get_random(1)
        self.game.visible_cards += CardList((TrainColor.WILD, 1))

        self.tree.next(DrawWildCardAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, TrainingNode))
        self.assertTrue(self.game.turn_state == TurnState.INIT)
        self.assertEqual(3, self.players[0].hand.number_of_cards())
        self.assertTrue(self.players[1].hand.has(CardList((TrainColor.WILD, 1))))
        self.assertEqual(5, self.players[1].hand.number_of_cards())

    def test_opponent_draw_visible_card_after_first_turn(self):
        self.__do_first_turn()
        self.tree.next(ClaimRouteAction(self.game, 2))
        self.game.visible_cards.get_random(1)
        self.game.visible_cards += CardList((TrainColor.GREEN, 1))

        self.tree.next(DrawVisibleCardAction(self.game, TrainColor.GREEN))

        self.assertTrue(isinstance(self.tree.current_node, OpponentNode))
        self.assertTrue(self.game.turn_state == TurnState.DRAWING_CARDS)
        self.assertEqual(3, self.players[0].hand.number_of_cards())
        self.assertTrue(self.players[1].hand.has(CardList((TrainColor.GREEN, 1))))
        self.assertEqual(5, self.players[1].hand.number_of_cards())

    def test_first_turn_over_after_player_and_opponent_select_destinations(self):
        self.__do_first_turn()

        self.assertEqual(0, self.game.current_player_index)
        self.assertTrue(isinstance(self.tree.current_node, TrainingNode))

        self.assertEqual(2, self.game.turn_count)
        self.assertTrue(self.game.turn_state == TurnState.INIT)
        self.assertTrue(self.game.state == GameState.PLAYING)

        self.assertEqual(3, len(self.players[0].uncompleted_destinations))
        self.assertEqual(3, len(self.players[1].uncompleted_destinations))

    def test_select_one_destination_then_end_first_turn(self):
        self.__do_training_node_turn()

        with self.assertRaises(ValueError):
            self.tree.next(DrawDestinationsAction(self.game))
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))
            self.tree.next(FinishSelectingDestinationsAction(self.game))

    def test_select_one_destination_then_end_turn_after_first_turn(self):
        self.__do_first_turn()
        self.__do_training_node_turn()

        self.tree.next(DrawDestinationsAction(self.game))
        self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.tree.next(FinishSelectingDestinationsAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, TrainingNode))
        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertEqual(GameState.PLAYING, self.game.state)
        self.assertEqual(0, self.game.current_player_index)

    def test_select_two_destinations_then_end_first_turn(self):
        self.__do_training_node_turn()

        self.tree.next(DrawDestinationsAction(self.game))
        for _ in range(2):
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.tree.next(FinishSelectingDestinationsAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, TrainingNode))
        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertEqual(GameState.PLAYING, self.game.state)
        self.assertEqual(0, self.game.current_player_index)
