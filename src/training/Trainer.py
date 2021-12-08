import os
import pickle
from datetime import datetime

from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.ActionUtility import ActionUtility
from src.training.GameNode import TrainingNode, OpponentNode
from src.training.GameTree import GameTree
from src.training.Regret import Regret
from src.training.Strategy import Strategy

import numpy as np

from src.training.StrategyStorage import StrategyStorage


class Trainer:
    def __init__(self):
        self.checkpoint_directory = "D:/Programming/TicketToRideMCCFR_TDD/checkpoints"
        self.tree = GameTree(Game([Player(), Player()], USMap()))
        self.strategy_storage = StrategyStorage()

    def train(self, iters):
        if iters <= 0:
            raise ValueError

        for i in range(iters):
            while self.tree.game.state != GameState.GAME_OVER:
                self.training_step()

            file_path = f"{self.checkpoint_directory}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.pkl"
            self.save_checkpoint(file_path)

            print(self.tree.game)

            self.tree = GameTree(Game([Player(), Player()], USMap()))

    # Strategy is based on which uncompleted_destinations the player has.
    # Each Training Node is allowed to learn while the game is being played.
    # Each Opponent Node will not learn during the game because it is impossible
    #   for the Training Node and Opponent Node to have the exact same uncompleted_destinations in the same game.
    def training_step(self):
        if not isinstance(self.tree.current_node, TrainingNode):
            self.tree.simulate_for_n_turns(1, self.strategy_storage)
            print("Opponent took their turn")

        if self.tree.game.state == GameState.GAME_OVER:
            return

        assert isinstance(self.tree.current_node, TrainingNode)

        # Determine the possible actions from the Training Node
        action_space = ActionSpace(self.tree.game)

        import random
        take_greedy_path = random.uniform(0, 1) < 0.3

        current_player = self.tree.game.current_player()
        training_strategy = self.strategy_storage.get(current_player.uncompleted_destinations)

        if take_greedy_path:
            # Choose the best valid action for the current set of uncompleted_destinations
            normalized_strategy = Strategy.normalize(training_strategy, action_space.to_np_array())
            action_id = int(np.argmax(normalized_strategy))
            chance = normalized_strategy[action_id]
            action = action_space.get_action_by_id(action_id)
            print(f"\nChose to take the GREEDY PATH {action} (normally {round(100*chance, 2)}% probability)\n")
        else:
            # Pick one using the blueprint strategy for this set of uncompleted_destinations
            action_id, chance = action_space.get_action_id(training_strategy)
            action = action_space.get_action_by_id(action_id)
            print(f"\nChose to {action} with {round(100*chance, 2)}% probability\n")

        # Determine the rewards (or utility) from each possible branch
        utils = ActionUtility.from_all_branches(self.tree.game, self.strategy_storage)
        # print("UTILITY FUNCTION")
        # print(utils)

        # Advance the game tree after game used for utils
        self.tree.next(action)

        # How much we regret not choosing a different branch
        regrets = Regret(utils).from_action_id(action_id)
        # print("REGRETS")
        # print(regrets)

        # Update the strategy
        new_strategy = Strategy.from_regrets(training_strategy, regrets)
        self.strategy_storage.set(current_player.uncompleted_destinations, new_strategy)
        print(f"\nUpdated Strategy for {sorted(current_player.uncompleted_destinations)}")
        # print(new_strategy)

    def save_checkpoint(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self.strategy_storage.strategies, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_latest_checkpoint(self):
        file_path = self.checkpoint_directory + "/" + os.listdir(self.checkpoint_directory)[-1]
        with open(file_path, "rb") as f:
            self.strategy_storage.strategies = pickle.load(f)
            print("CHECKPOINT LOADED", file_path)
            print("LENGTH: ", len(self.strategy_storage))
            print()

    def display_strategy(self):
        sorted_strategies = sorted(self.strategy_storage.strategies.items(), key=lambda x: x[0])
        for index, (key, strategy) in enumerate(sorted_strategies):
            if index > 100:
                break

            print("STRATEGY FOR", key)
            normalized = Strategy.normalize(strategy)
            pairs = {}

            for idx, val in enumerate(normalized):
                action = ActionSpace(self.tree.game).get_action_by_id(idx)
                pairs[str(action)] = "{:.3f}%".format(round(100 * val, 3))

            sorted_items = sorted(pairs.items(), key=lambda x: x[1])
            print("Most Useful:", sorted_items[-3:])
            print("Least Useful:", sorted_items[:3])
            print()
