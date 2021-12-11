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
from src.training.GameNode import TrainingNode, OpponentNode
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

        # Temporary to allow players to experience the full action space
        self.tree.game.available_destinations = [_id for _id in self.tree.game.unclaimed_destinations]

        # Temporary to allow players to experience the full action space
        for player in self.tree.game.players:
            player.hand = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])

        # Temporary to allow players to experience the full action space
        self.tree.game.visible_cards = CardList.from_numbers([12, 12, 12, 12, 12, 12, 12, 12, 14])

        if not isinstance(self.tree.current_node, TrainingNode):
            self.tree.greedy_simulation_for_n_turns(1, self.strategy_storage)
            # print("Opponent took their turn")
            # print(self.tree.game)

        if self.tree.game.state == GameState.GAME_OVER:
            return

        assert isinstance(self.tree.current_node, TrainingNode)

        # Determine the possible actions from the Training Node
        action_space = ActionSpace(self.tree.game)
        print("ACTION SPACE")
        print(action_space)

        import random
        take_greedy_path = random.uniform(0, 1) < 0.5

        current_player = self.tree.game.current_player()
        training_strategy = self.strategy_storage.get(current_player.uncompleted_destinations)
        print("\nLOADED TRAINING STRATEGY for", current_player.uncompleted_destinations)
        print(training_strategy)

        if take_greedy_path:
            # Choose the best valid action for the current set of uncompleted_destinations
            normalized_strategy = Strategy.normalize(training_strategy, action_space.to_np_array())
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
        self.strategy_storage.set(current_player.uncompleted_destinations, new_strategy)
        print(f"\nUpdated Strategy for {sorted(current_player.uncompleted_destinations)}")

        # Detect if the player is stuck with nothing to do (i.e. no more cards left to draw)
        if chance == 0:
            self.tree.game.turn_state = TurnState.FINISHED
            self.tree.current_node = self.tree.current_node.pass_turn()
        else:
            # Advance the game tree with the chosen action
            self.tree.next(action)

    def save_checkpoint(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self.strategy_storage.strategies, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_latest_checkpoint(self):
        file_path = self.checkpoint_directory + "/" + os.listdir(self.checkpoint_directory)[-1]
        with open(file_path, "rb") as f:
            self.strategy_storage.strategies = pickle.load(f)
            print("\nCHECKPOINT LOADED", file_path)
            print("LENGTH: ", len(self.strategy_storage))
            print()

        # self.check_for_missing_keys()

    def check_for_missing_keys(self):
        # The strategies still missing from every single two-destination combo
        missing_keys = set()
        missing_two_dest_keys = set()
        missing_three_dest_keys = set()

        for id1, d1 in USMap().destinations.items():
            if self.strategy_storage.get({id1: d1}).tolist() == Strategy.random(141).tolist():
                missing_keys.add(str({id1: d1}))

            for id2, d2 in USMap().destinations.items():
                if self.strategy_storage.get({id1: d1, id2: d2}).tolist() == Strategy.random(141).tolist():
                    missing_two_dest_keys.add(str(sorted({id1: d1, id2: d2})))

                for id3, d3 in USMap().destinations.items():
                    if self.strategy_storage.get({id1: d1, id2: d2, id3: d3}).tolist() == Strategy.random(141).tolist():
                        missing_three_dest_keys.add(str(sorted({id1: d1, id2: d2, id3: d3})))
        print("missing single-destination keys:", len(missing_keys))
        print(missing_keys)

        print("missing two-destination keys:", len(missing_two_dest_keys))
        print(missing_two_dest_keys)

        print("missing three-destination keys:", len(missing_three_dest_keys))
        print(missing_three_dest_keys)

    def display_strategy(self):
        for index, (key, strategy) in enumerate(self.strategy_storage.strategies.items()):
            if index > 50:
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

        print("SHOWING THE FIRST 50 of", len(self.strategy_storage))
