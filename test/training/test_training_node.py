import unittest

from src.actions.ClaimRouteAction import ClaimRouteAction
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.actions.DrawWildCardAction import DrawWildCardAction
from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TrainColor import TrainColor
from src.game.enums.TurnState import TurnState
from src.training.GameNode import Player2Node, Player1Node
from src.training.GameTree import GameTree
from src.game.CardList import CardList
from src.training.StrategyStorage import StrategyStorage


class TrainingNodeTest(unittest.TestCase):
    def setUp(self):
        self.players = [Player(), Player()]
        self.game = Game(self.players, USMap())
        self.tree = GameTree(self.game)

    def __do_first_turn(self):
        for _ in range(2):
            self.tree.next(DrawDestinationsAction(self.game))
            for _ in range(3):
                self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

    def test_first_turn_switch_after_select_destinations(self):
        self.tree.next(DrawDestinationsAction(self.game))
        for _ in range(3):
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertEqual(GameState.FIRST_ROUND, self.game.state)
        self.assertEqual(1, self.game.current_player_index)
        self.assertEqual(1, self.game.turn_count)

    def test_select_one_destination_then_end_first_turn(self):
        with self.assertRaises(ValueError):
            self.tree.next(DrawDestinationsAction(self.game))
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))
            self.tree.next(FinishSelectingDestinationsAction(self.game))

    def test_select_one_destination_then_end_turn_after_first_turn(self):
        self.__do_first_turn()

        self.tree.next(DrawDestinationsAction(self.game))
        self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.tree.next(FinishSelectingDestinationsAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertEqual(GameState.PLAYING, self.game.state)
        self.assertEqual(1, self.game.current_player_index)

    def test_select_two_destinations_then_end_first_turn(self):
        self.tree.next(DrawDestinationsAction(self.game))
        for _ in range(2):
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.tree.next(FinishSelectingDestinationsAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertEqual(GameState.FIRST_ROUND, self.game.state)
        self.assertEqual(1, self.game.current_player_index)

    def test_draw_random_card_after_first_turn(self):
        self.__do_first_turn()

        self.tree.next(DrawRandomCardAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, Player1Node))
        self.assertTrue(self.game.turn_state == TurnState.DRAWING_CARDS)
        self.assertEqual(5, self.game.players[0].hand.number_of_cards())
        self.assertEqual(4, self.game.players[1].hand.number_of_cards())

    def test_draw_wild_after_first_turn(self):
        self.__do_first_turn()
        self.game.visible_cards.get_random(1)
        self.game.visible_cards += CardList((TrainColor.WILD, 1))

        self.tree.next(DrawWildCardAction(self.game))

        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
        self.assertTrue(self.game.turn_state == TurnState.INIT)
        self.assertEqual(5, self.players[0].hand.number_of_cards())
        self.assertTrue(self.players[0].hand.has(CardList((TrainColor.WILD, 1))))
        self.assertEqual(4, self.players[1].hand.number_of_cards())

    def test_draw_visible_card_after_first_turn(self):
        self.__do_first_turn()
        self.game.visible_cards.get_random(1)
        self.game.visible_cards += CardList((TrainColor.GREEN, 1))

        self.tree.next(DrawVisibleCardAction(self.game, TrainColor.GREEN))

        self.assertTrue(isinstance(self.tree.current_node, Player1Node))
        self.assertTrue(self.game.turn_state == TurnState.DRAWING_CARDS)
        self.assertEqual(5, self.players[0].hand.number_of_cards())
        self.assertTrue(self.players[0].hand.has(CardList((TrainColor.GREEN, 1))))
        self.assertEqual(4, self.players[1].hand.number_of_cards())

    def test_draw_destinations_after_first_turn(self):
        self.__do_first_turn()
        self.tree.next(DrawDestinationsAction(self.game))

        self.assertEqual(3, len(self.game.available_destinations))
        self.assertEqual(TurnState.SELECTING_DESTINATIONS, self.game.turn_state)
        self.assertTrue(isinstance(self.tree.current_node, Player1Node))
        self.assertEqual(3, len(self.players[0].uncompleted_destinations))

    def test_draw_and_select_destinations_after_first_turn(self):
        self.__do_first_turn()
        self.tree.next(DrawDestinationsAction(self.game))

        for _ in range(3):
            self.tree.next(SelectDestinationAction(self.game, self.game.available_destinations[0]))

        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
        self.assertEqual([], self.game.available_destinations)
        self.assertEqual(6, len(self.players[0].uncompleted_destinations))

    def test_claim_route_action_after_first_turn(self):
        self.__do_first_turn()

        self.tree.next(ClaimRouteAction(self.game, 2))

        self.assertEqual(TurnState.INIT, self.game.turn_state)
        self.assertTrue(isinstance(self.tree.current_node, Player2Node))
        self.assertFalse(2 in self.game.unclaimed_routes)
        self.assertTrue(2 in self.players[0].routes)
        self.assertEqual(1, self.players[0].points)
        self.assertEqual(44, self.players[0].trains)

    def test_init_info_sets(self):
        self.game.players[0].hand = CardList((TrainColor.BLUE, 4))
        tree = GameTree(self.game)

        self.assertEqual(f"p1_start_cards_{self.game.players[0].hand} ",
                         tree.current_node.cumulative_information_sets[0])

        self.assertEqual(f"p2_start_cards_{self.game.players[1].hand} ",
                         tree.current_node.cumulative_information_sets[1])

    def test_node_id_added_to_when_actions_executed_training_player1(self):
        self.tree.game.unclaimed_destinations = {2: USMap().destinations.get(2)}
        self.tree.training_node_type = Player1Node
        p1_starting_cards = self.tree.current_node.cumulative_information_sets[0]
        p2_starting_cards = self.tree.current_node.cumulative_information_sets[1]

        self.tree.next(DrawDestinationsAction(self.tree.game))
        self.assertEqual("p1_draw_dest", self.tree.current_node.current_turn_information_set)

        self.tree.next(SelectDestinationAction(self.tree.game, 2))
        self.assertEqual(p1_starting_cards + "p1_draw_dest_AND_select_CALGARY_to_SALT_LAKE_CITY ",
                         self.tree.current_node.cumulative_information_sets[0])

        # Player 1 Turn Over, Player 2's Information Set

        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection ", self.tree.current_node.get_cumulative_information_set())

        self.tree.game.unclaimed_destinations = {i: USMap().destinations.get(i) for i in [3, 4, 5]}
        self.tree.next(DrawDestinationsAction(self.tree.game))
        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection p2_draw_dest",
                         self.tree.current_node.get_cumulative_information_set())
        self.assertEqual("p2_draw_dest", self.tree.current_node.current_turn_information_set)

        self.tree.next(SelectDestinationAction(self.tree.game, 4))
        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection p2_draw_dest_AND_select_CHICAGO_to_SANTA_FE",
                         self.tree.current_node.get_cumulative_information_set())
        self.assertEqual("p2_draw_dest_AND_select_CHICAGO_to_SANTA_FE",
                         self.tree.current_node.current_turn_information_set)

        self.tree.next(SelectDestinationAction(self.tree.game, 3))
        self.assertEqual(p2_starting_cards +
                         "p1_hidden_dest_selection p2_draw_dest_AND_select_CHICAGO_to_NEW_ORLEANS_AND_CHICAGO_to_SANTA_FE",
                         self.tree.current_node.get_cumulative_information_set())

        self.assertEqual("p2_draw_dest_AND_select_CHICAGO_to_NEW_ORLEANS_AND_CHICAGO_to_SANTA_FE",
                         self.tree.current_node.current_turn_information_set)

        self.tree.next(FinishSelectingDestinationsAction(self.tree.game))
        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection " +
                         "p2_draw_dest_AND_select_CHICAGO_to_NEW_ORLEANS_AND_CHICAGO_to_SANTA_FE ",
                         self.tree.current_node.cumulative_information_sets[1])

        # Player 2 Turn Over, Player 1's Information Set

        self.assertEqual(p1_starting_cards + "p1_draw_dest_AND_select_CALGARY_to_SALT_LAKE_CITY p2_hidden_dest_selection ",
                         self.tree.current_node.get_cumulative_information_set())

    def test_node_id_added_to_when_actions_executed_training_player2(self):
        self.tree.game.unclaimed_destinations = {2: USMap().destinations.get(2)}
        self.tree.training_node_type = Player2Node

        p1_starting_cards = self.tree.current_node.cumulative_information_sets[0]
        p2_starting_cards = self.tree.current_node.cumulative_information_sets[1]

        self.tree.simulate_for_n_turns(1, StrategyStorage())

        # Player 1 Turn Over

        self.assertEqual(p1_starting_cards + "p1_draw_dest_AND_select_CALGARY_to_SALT_LAKE_CITY ",
                         self.tree.current_node.cumulative_information_sets[0])
        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection ", self.tree.current_node.get_cumulative_information_set())

        self.tree.game.unclaimed_destinations = {3: USMap().destinations.get(3)}
        self.tree.next(DrawDestinationsAction(self.tree.game))
        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection p2_draw_dest",
                         self.tree.current_node.get_cumulative_information_set())
        self.assertEqual("p2_draw_dest", self.tree.current_node.current_turn_information_set)

        self.tree.next(SelectDestinationAction(self.tree.game, 3))

        # Player 2 Turn Over
        self.assertEqual(p2_starting_cards + "p1_hidden_dest_selection p2_draw_dest_AND_select_CHICAGO_to_NEW_ORLEANS ",
                         self.tree.current_node.cumulative_information_sets[1])
        self.assertEqual(p1_starting_cards + "p1_draw_dest_AND_select_CALGARY_to_SALT_LAKE_CITY p2_hidden_dest_selection ",
                         self.tree.current_node.get_cumulative_information_set())

    def test_cumulative_information_sets_are_not_the_same(self):
        self.tree.simulate_for_n_turns(4, StrategyStorage())
        set1 = self.tree.current_node.cumulative_information_sets[0]
        set2 = self.tree.current_node.cumulative_information_sets[1]
        print(set1)
        print(set2)

        self.assertNotEqual(set1, set2)

    def test_claim_route_action_added_to_id_when_executed(self):
        self.tree.game.players[0].hand = CardList((TrainColor.WILD, 6))
        self.tree.simulate_for_n_turns(2, StrategyStorage())

        self.assertEqual(GameState.PLAYING, self.tree.game.state)
        self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

        action = ClaimRouteAction(self.tree.game, 5)
        expected = self.tree.current_node.get_cumulative_information_set() + f"p1_{str(action)} "

        self.tree.next(action)

        # Player 1 Turn Over

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[0])

    def test_claim_route_action_added_to_id_when_executed_player2(self):
        self.tree.simulate_for_n_turns(3, StrategyStorage())
        self.tree.game.players[1].hand = CardList((TrainColor.WILD, 6))

        self.assertEqual(GameState.PLAYING, self.tree.game.state)
        self.assertNotEqual("", self.tree.current_node.cumulative_information_sets[1])

        action = ClaimRouteAction(self.tree.game, 5)
        expected = self.tree.current_node.cumulative_information_sets[1] + f"p2_{str(action)} "

        self.tree.next(action)

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[1])

    def test_draw_random_action_for_current_player_is_the_color_of_the_card(self):
        self.tree.simulate_for_n_turns(2, StrategyStorage())
        self.assertEqual(GameState.PLAYING, self.tree.game.state)
        self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

        self.tree.game.deck = CardList((TrainColor.GREEN, 1))
        action = DrawRandomCardAction(self.tree.game)

        self.tree.next(action)
        self.assertEqual("p1_draw_GREEN", self.tree.current_node.current_turn_information_set)

        self.tree.game.deck = CardList((TrainColor.BLUE, 1))
        action = DrawRandomCardAction(self.tree.game)

        expected = self.tree.current_node.cumulative_information_sets[0] + "p1_draw_BLUE_AND_draw_GREEN "

        self.tree.next(action)

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[0])

    def test_draw_random_not_added_until_end_of_turn_player2(self):
        self.tree.simulate_for_n_turns(3, StrategyStorage())

        self.assertEqual(GameState.PLAYING, self.tree.game.state)
        self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

        expected = self.tree.current_node.cumulative_information_sets[0]

        self.tree.next(DrawRandomCardAction(self.tree.game))

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[0])

        self.tree.next(DrawRandomCardAction(self.tree.game))

        self.assertEqual(expected + "p2_draw_RANDOM_AND_draw_RANDOM ", self.tree.current_node.cumulative_information_sets[0])

    def test_draw_wild_action_added_to_id_when_executed_player1(self):
        self.tree.simulate_for_n_turns(2, StrategyStorage())
        self.tree.game.visible_cards = CardList((TrainColor.WILD, 1))

        self.assertEqual(GameState.PLAYING, self.tree.game.state)
        self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

        action = DrawWildCardAction(self.tree.game)
        expected = self.tree.current_node.cumulative_information_sets[0] + f"p1_{str(action)} "

        self.tree.next(action)

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[0])

    def test_draw_wild_action_added_to_id_when_executed_player2(self):
        self.tree.simulate_for_n_turns(3, StrategyStorage())
        self.tree.game.visible_cards = CardList((TrainColor.WILD, 1))

        self.assertEqual(GameState.PLAYING, self.tree.game.state)
        self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

        action = DrawWildCardAction(self.tree.game)
        expected = self.tree.current_node.cumulative_information_sets[1] + f"p2_{str(action)} "

        self.tree.next(action)

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[1])

    def test_draw_visible_card_not_added_to_until_end_of_turn_player1(self):
        for color in TrainColor:
            if color == TrainColor.WILD:
                continue

            self.tree = GameTree(Game([Player(), Player()], USMap()))
            self.tree.simulate_for_n_turns(2, StrategyStorage())
            self.tree.game.visible_cards = CardList((color, 1))

            self.assertEqual(GameState.PLAYING, self.tree.game.state)
            self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

            action = DrawVisibleCardAction(self.tree.game, color)
            expected = f"p1_{str(action)}"

            self.tree.next(action)

            self.assertEqual(expected, self.tree.current_node.current_turn_information_set)

    def test_draw_visible_card_action_added_to_id_when_executed_player2(self):
        for color in TrainColor:
            if color == TrainColor.WILD:
                continue

            self.tree = GameTree(Game([Player(), Player()], USMap()))
            self.tree.simulate_for_n_turns(3, StrategyStorage())
            self.tree.game.visible_cards = CardList((color, 1))

            self.assertEqual(GameState.PLAYING, self.tree.game.state)
            self.assertNotEqual("", self.tree.current_node.get_cumulative_information_set())

            action = DrawVisibleCardAction(self.tree.game, color)
            expected = self.tree.current_node.cumulative_information_sets[1]

            self.tree.next(action)

            self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[1])

    def test_draw_two_randoms_that_are_both_wild(self):
        self.tree.simulate_for_n_turns(2, StrategyStorage())

        self.assertEqual(GameState.PLAYING, self.tree.game.state)

        self.tree.game.deck = CardList((TrainColor.WILD, 2))
        expected = self.tree.current_node.get_cumulative_information_set() + "p1_draw_WILD_AND_draw_WILD "

        self.tree.next(DrawRandomCardAction(self.tree.game))
        self.tree.next(DrawRandomCardAction(self.tree.game))

        # Player 1 turn Over

        self.assertEqual(expected, self.tree.current_node.cumulative_information_sets[0])
        self.assertEqual("", self.tree.current_node.current_turn_information_set)

    def test_draw_two_randoms_that_are_both_wild_hidden_to_other_player(self):
        self.tree.simulate_for_n_turns(2, StrategyStorage())

        self.assertEqual(GameState.PLAYING, self.tree.game.state)

        self.tree.game.deck = CardList((TrainColor.WILD, 2))
        expected = self.tree.current_node.cumulative_information_sets[1] + "p1_draw_RANDOM_AND_draw_RANDOM "

        self.tree.next(DrawRandomCardAction(self.tree.game))
        self.tree.next(DrawRandomCardAction(self.tree.game))

        # Player 1 turn Over

        self.assertEqual(expected, self.tree.current_node.get_cumulative_information_set())
        self.assertEqual("", self.tree.current_node.current_turn_information_set)

    def test_draw_random_and_visible_revealed_for_current_player(self):
        self.tree.simulate_for_n_turns(2, StrategyStorage())

        self.assertEqual(GameState.PLAYING, self.tree.game.state)

        self.tree.game.visible_cards = CardList((TrainColor.YELLOW, 1))
        self.tree.game.deck = CardList((TrainColor.GREEN, 1))
        expected_player1_info_set = \
            self.tree.current_node.cumulative_information_sets[0] + "p1_draw_GREEN_AND_draw_YELLOW "

        expected_player2_info_set = \
            self.tree.current_node.cumulative_information_sets[1] + "p1_draw_RANDOM_AND_draw_YELLOW "

        self.tree.next(DrawRandomCardAction(self.tree.game))
        self.tree.next(DrawVisibleCardAction(self.tree.game, TrainColor.YELLOW))

        # Player 1 turn Over

        self.assertEqual(expected_player1_info_set, self.tree.current_node.cumulative_information_sets[0])
        self.assertEqual(expected_player2_info_set, self.tree.current_node.cumulative_information_sets[1])
