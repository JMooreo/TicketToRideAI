from __future__ import annotations

import numpy as np
from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.game.enums.TurnState import TurnState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import Player1Node, GameNode
from src.training.InformationSet import InformationSet
from src.training.ObservationSpace import ObservationSpace


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

    def simulate_for_n_turns(self, num_turns, agent):
        action_space = ActionSpace(self.game)
        observation_space = ObservationSpace(self.game)
        for _ in range(num_turns):
            if self.game.state == GameState.GAME_OVER:
                break

            node_type = self.current_node.__class__

            while isinstance(self.current_node, node_type):
                if sum(action_space.valid_action_mask()) == 0:
                    self.current_node = self.current_node.pass_turn()
                    break

                action_id = agent.act(observation_space.to_np_array(), action_space.valid_action_mask())
                action = action_space.get_action_by_id(action_id)

                self.next(action)

    def simulate_until_game_over(self, agent):
        while self.game.state != GameState.GAME_OVER:
            self.simulate_for_n_turns(1, agent)

    def __initialize_info_sets(self):
        for player_idx, player in enumerate(self.game.players):
            self.current_node.information_set = InformationSet.from_game(self.game, player_idx)
