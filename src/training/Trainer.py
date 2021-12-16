import os
import pickle
from datetime import datetime

from src.game.CardList import CardList
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.ActionUtility import ActionUtility
from src.training.GameNode import Player1Node, Player2Node
from src.training.GameTree import GameTree
from src.training.Regret import Regret
from src.training.Strategy import Strategy

import numpy as np

from src.training.StrategyStorage import StrategyStorage


class Trainer:
    def __init__(self):
        self.checkpoint_directory = "D:/Programming/TicketToRideMCCFR_TDD/single-destination-checkpoints"
        self.tree = GameTree(Game([Player(), Player()], USMap()))
        self.strategy_storage = StrategyStorage()

    def train(self, iters):
        if iters <= 0:
            raise ValueError

        for i in range(iters):
            for node_type_to_train in [Player1Node, Player2Node]:
                self.tree = GameTree(Game([Player(), Player()], USMap()))
                self.tree.training_node_type = node_type_to_train

                while self.tree.game.state != GameState.GAME_OVER:
                    self.training_step(node_type_to_train)

                file_path = f"{self.checkpoint_directory}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.pkl"
                self.save_checkpoint(file_path)

                print(self.tree.game)

    def training_step(self, node_type_to_train):
        if not isinstance(self.tree.current_node, node_type_to_train):
            self.tree.simulate_for_n_turns(1, self.strategy_storage)
            # print("Opponent took their turn")
            # print(self.tree.game)

        if self.tree.game.state == GameState.GAME_OVER:
            return

        assert isinstance(self.tree.current_node, node_type_to_train)

        # Determine the possible actions from the Training Node
        action_space = ActionSpace(self.tree.game)
        action_space_np = action_space.to_np_array()

        # Detect if the player cant do anything
        if sum(action_space_np) == 0:
            self.tree.game.turn_state = TurnState.FINISHED
            self.tree.current_node = self.tree.current_node.pass_turn()
            return

        print(f"ACTION SPACE for")
        print(action_space)

        import random
        take_greedy_path = random.uniform(0, 1) < 0.3

        info_set = self.tree.current_node.information_set
        training_strategy = self.strategy_storage.get_node_strategy(info_set)
        print(f"\nLOADED TRAINING STRATEGY for \"{info_set}\"")
        print(training_strategy)

        if take_greedy_path:
            # Choose the best valid action for the current set of uncompleted_destinations
            normalized_strategy = Strategy.normalize(training_strategy, action_space_np)
            action_id = int(np.argmax(normalized_strategy))
            chance = normalized_strategy[action_id]
            action = action_space.get_action_by_id(action_id)
            print(f"\nChose to take the GREEDY PATH {action} (normally {round(100*chance, 2)}% probability)")
        else:
            # Pick one using the blueprint strategy for this set of uncompleted_destinations
            action_id, chance = action_space.get_action_id(training_strategy)
            action = action_space.get_action_by_id(action_id)
            print(f"\nChose to {action} with {round(100*chance, 2)}% probability")

        # Determine the rewards (or utility) from each possible branch
        utils = ActionUtility.from_all_branches(self.tree.game, self.strategy_storage)
        # print("UTILITY FUNCTION")
        # print(utils)

        # How much we regret not choosing a different branch
        regrets = Regret(utils, 1).from_action_id(action_id)
        highest_regret = int(np.argmax(regrets))
        # print("REGRETS")
        # print(regrets)
        if highest_regret > 0:
            print("\nPlayer regrets not choosing", action_space.get_action_by_id(highest_regret))

        # Update the strategy
        new_strategy = Strategy.from_regrets(training_strategy, regrets)
        cumulative_info_set = self.tree.current_node.information_set
        self.strategy_storage.set(cumulative_info_set, new_strategy)
        print(f"\nUpdated Strategy for \"{cumulative_info_set}\"")

        # Advance the game tree with the chosen action
        self.tree.next(action)

    def save_checkpoint(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self.strategy_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_latest_checkpoint(self):
        file_path = self.checkpoint_directory + "/" + os.listdir(self.checkpoint_directory)[-1]
        with open(file_path, "rb") as f:
            self.strategy_storage = pickle.load(f)
            print("\nCHECKPOINT LOADED", file_path)
            print("LENGTH: ", len(self.strategy_storage))
            print()

        # self.check_for_missing_keys()

    # def display_strategy(self):
    #     for index, (key, strategy) in enumerate(self.strategy_storage.strategies.items()):
    #         if index > 50:
    #             break
    #
    #         print("STRATEGY FOR", key)
    #         normalized = Strategy.normalize(strategy)
    #         pairs = {}
    #
    #         for idx, val in enumerate(normalized):
    #             action = ActionSpace(self.tree.game).get_action_by_id(idx)
    #             pairs[str(action)] = "{:.3f}%".format(round(100 * val, 3))
    #
    #         sorted_items = sorted(pairs.items(), key=lambda x: x[1])
    #         print("Most Useful:", sorted_items[-3:])
    #         print("Least Useful:", sorted_items[:3])
    #         print()
    #
    #     print("SHOWING THE FIRST 50 of", len(self.strategy_storage))
