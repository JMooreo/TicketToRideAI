from __future__ import annotations

import random
import numpy as np
from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import Player1Node, GameNode
from src.training.Strategy import Strategy
from src.training.StrategyStorage import StrategyStorage


class GameTree:
    def __init__(self, game: Game):
        self.game = game
        self.current_node: GameNode = Player1Node(self.game)
        self.training_node_type = Player1Node

        self.__initialize_info_sets()

    def next(self, action: Action):
        if action is None or not action.is_valid():
            raise ValueError(f"The action could not be executed because it was invalid.\n" +
                             f"Action: {action}\n" +
                             str(self.game))

        self.current_node = self.current_node.next(action)

    def simulate_for_n_turns(self, num_turns, strategy_storage: StrategyStorage):
        action_space = ActionSpace(self.game)
        for _ in range(num_turns):
            if self.game.state == GameState.GAME_OVER:
                break

            node_type = self.current_node.__class__

            while isinstance(self.current_node, node_type):
                strategy = strategy_storage.get_node_strategy(self.current_node.get_cumulative_information_set())
                action_id, chance = action_space.get_action_id(strategy)
                action = action_space.get_action_by_id(action_id)
                player_idx = self.game.current_player_index

                self.next(action)
                strategy_storage.increment_average_strategy(player_idx, action_id)

    def simulate_until_game_over(self, strategy_storage: StrategyStorage):
        while self.game.state != GameState.GAME_OVER:
            self.simulate_for_n_turns(1, strategy_storage)

    def greedy_simulation_for_n_turns(self, num_turns, strategy_storage: StrategyStorage):
        action_space = ActionSpace(self.game)

        for _ in range(num_turns):
            if self.game.state == GameState.GAME_OVER:
                break

            node_type = self.current_node.__class__

            while isinstance(self.current_node, node_type):
                strategy = strategy_storage.get_node_strategy(self.current_node.get_cumulative_information_set())
                normalized_strategy = Strategy.normalize(strategy, action_space.to_np_array())
                if sum(normalized_strategy) == 0:
                    # print(self.game)
                    # print(f"Player {self.game.current_player_index + 1} couldn't take an action, so they skipped.")
                    self.game.turn_state = TurnState.FINISHED
                    self.current_node = self.current_node.pass_turn()
                    continue

                take_greedy_path = random.uniform(0, 1) < 0.3

                if take_greedy_path:
                    # Choose the best valid action for the current set of uncompleted_destinations
                    action_id = int(np.argmax(normalized_strategy))
                    chance = normalized_strategy[action_id]
                    action = action_space.get_action_by_id(action_id)
                else:
                    # Explore an action from the current strategy
                    action_id, chance = action_space.get_action_id(strategy)
                    action = action_space.get_action_by_id(action_id)

                self.next(action)

    def greedy_simulation_until_game_over(self, strategy_storage: StrategyStorage):
        while self.game.state != GameState.GAME_OVER:
            self.greedy_simulation_for_n_turns(1, strategy_storage)

    def __initialize_info_sets(self):
        for player_idx, player in enumerate(self.game.players):
            self.current_node.cumulative_information_sets[player_idx] = f"p{player_idx + 1}_start_cards_{player.hand} "
