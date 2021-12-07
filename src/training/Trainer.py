import os
from datetime import datetime

from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.ActionUtility import ActionUtility
from src.training.GameNode import TrainingNode
from src.training.GameTree import GameTree
from src.training.Regret import Regret
from src.training.Strategy import Strategy

import numpy as np


class Trainer:
    def __init__(self):
        self.checkpoint_directory = "D:/Programming/TicketToRideMCCFR_TDD/checkpoints"
        self.tree = self.new_game_tree()
        self.strategy = Strategy.random(len(ActionSpace(self.tree.game)))

    def new_game_tree(self):
        # Skip past the first "Draw Destinations" action
        # because there are no other choices.
        tree = GameTree(Game([Player(), Player()], USMap()))
        tree.next(DrawDestinationsAction(tree.game))
        return tree

    def train(self, iters):
        if iters <= 0:
            raise ValueError

        for i in range(iters):
            while self.tree.game.state != GameState.GAME_OVER:
                self.training_step()

            file_path = f"{self.checkpoint_directory}/checkpoint-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.txt"
            with open(file_path, "w") as f:
                np.savetxt(f, self.strategy)

            # Reset the game tree. This happens after the simulation to allow faster tests.
            self.tree = self.new_game_tree()

            print(self.tree.game)

    def training_step(self):
        if not isinstance(self.tree.current_node, TrainingNode):
            self.tree.simulate_for_n_turns(1)

        if self.tree.game.state == GameState.GAME_OVER:
            return

        # Determine the possible actions
        action_space = ActionSpace(self.tree.game)
        print("ACTION SPACE")
        print(action_space)

        # Pick one using the blueprint strategy
        action_id, chance = action_space.get_action_id(self.strategy)
        action = action_space.get_action_by_id(action_id)
        print("ACTION")
        print(f"Chose to {action} with {round(100*chance, 2)}% probability\n")

        # Determine the rewards (or utility) from each possible branch
        utils = ActionUtility.from_all_branches(self.tree.game)
        print("UTILITY FUNCTION")
        print(utils)

        # How much we regret not choosing a different branch
        regrets = Regret(utils).from_action_id(action_id)
        print("REGRETS")
        print(regrets)

        # Update the strategy
        self.strategy = Strategy.from_regrets(self.strategy, regrets)
        print("NEW STRATEGY")
        print(self.strategy)

        # Advance the game tree
        self.tree.next(action)

        # Show the state of the game
        print(self.tree.game)

    def load_latest_checkpoint(self):
        try:
            latest_checkpoint = self.checkpoint_directory + "/" + os.listdir(self.checkpoint_directory)[-1]
            print("LOADED", latest_checkpoint)
            self.strategy = np.loadtxt(latest_checkpoint)
        except IndexError:
            print("COULDNT LOAD STRATEGY, using a random strategy instead")
            self.strategy = Strategy.random(len(ActionSpace(self.tree.game)))
