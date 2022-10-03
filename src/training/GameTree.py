from __future__ import annotations

from typing import List

from src.actions.Action import Action
from src.game.Game import Game
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameNode import Player1Node, GameNode
from src.training.ObservationSpace import ObservationSpace


class GameTree:
    def __init__(self, game: Game):
        self.game = game
        self.current_node: GameNode = Player1Node(self.game)
        self.training_node_type = Player1Node

    def next(self, action: Action):
        if action is None or not action.is_valid():
            raise ValueError(f"The action could not be executed because it was invalid.\n" +
                             f"Action: {action}\n" +
                             str(self.game))

        self.current_node = self.current_node.next(action)

    def simulate_for_n_turns(self, num_turns, agents: List, debug=False):
        from actors.Agent import Agent
        from Environments.TTREnv import TTREnv

        action_space = ActionSpace(self.game)

        env = TTREnv(self, action_space, ObservationSpace(self.game))

        for _ in range(num_turns):
            if self.game.state == GameState.GAME_OVER:
                break

            node_type = self.current_node.__class__
            agent: Agent = agents[self.game.current_player_index]

            if debug:
                print(f"\nNew Turn {agent}")

            while isinstance(self.current_node, node_type):
                if sum(action_space.valid_action_mask()) == 0:
                    self.current_node = self.current_node.pass_turn()
                    break

                action_id = agent.act(env)
                action = action_space.get_action_by_id(action_id)

                if debug:
                    print(f"Player {self.game.current_player_index + 1} Taking Action {action}")

                self.next(action)

    def simulate_until_game_over(self, agents: List, debug=False):
        if len(agents) != len(self.game.players):
            raise ValueError("Number of agents must match the number of players! "
                             f"{len(agents)} != {len(self.game.players)}")

        while self.game.state != GameState.GAME_OVER:
            self.simulate_for_n_turns(1, agents, debug)
