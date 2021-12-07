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


class Trainer:
    def __init__(self):
        self.tree = GameTree(Game([Player(), Player()], USMap()))
        self.strategy = Strategy.random(len(ActionSpace(self.tree.game)))

    def train(self, iters):
        if iters <= 0:
            raise ValueError

        for i in range(iters):
            while self.tree.game.state != GameState.GAME_OVER:
                self.training_step()

    def training_step(self):
        # There is no need to train on the first action because it is the only action.
        # (Draw Destinations) so a good improvement would be to skip that.

        if not isinstance(self.tree.current_node, TrainingNode):
            self.tree.simulate_for_n_turns(1)

        action_space = ActionSpace(self.tree.game)
        print("ACTION SPACE")
        print(action_space)
        action_id, chance = action_space.get_action_id(self.strategy)
        action = action_space.get_action_by_id(action_id)
        print("ACTION")
        print(f"Chose to {action} with {round(100*chance, 2)}% probability\n")
        utils = ActionUtility.from_all_branches(self.tree.game)
        print("UTILITY FUNCTION")
        print(utils)
        regrets = Regret(utils).from_action_id(action_id)
        print("REGRETS")
        print(regrets)
        self.strategy = Strategy.normalize_from_regrets(self.strategy, regrets)
        print("NEW STRATEGY")
        print(self.strategy)

        # Take the action we said that we would
        self.tree.next(action)

        print(self.tree.game)
