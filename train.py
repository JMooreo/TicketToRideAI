import os
import sys

import gym
import wandb
from actors.RandomAgent import RandomAgent

from src.Environments.TTREnv import TTREnv

if __name__ == "__main__":
    env = TTREnv()
    env.tree.simulate_until_game_over([RandomAgent(), RandomAgent()])
    print()
    # wandb.init(project="ticket-to-ride-ai", entity="jmooreo")
    # wandb.config = {
    #     'epsilon_start': 1,
    #     'epsilon_end': 0.05,
    #     'epsilon_decay_period': 1000,
    #     'target_update_freq': 1000,
    #     'logging_freq': 10,
    #     'learning_rate': 0.01,
    #     'epochs': 10000
    # }
    #
    # args = {item.split("=")[0][2:]: float(item.split("=")[1]) for item in sys.argv[1:]}
    #
    # if len(args) > 0:
    #     wandb.config = args
    #
    # checkpoint_dir = "D:/Programming/TicketToRideMCCFR_TDD/checkpoints/CartPole-DeepQ"
    #
    # # Ensure directory exists
    # try:
    #     os.mkdir(checkpoint_dir)
    # except:
    #     pass
    #
    # env = gym.make("CartPole-v0")
    #
    # algorithm = SingleActorDeepQLearningAlgorithm(env, OneLayerTanh, wandb, **wandb.config)
    # algorithm.load_latest_checkpoint(checkpoint_dir)
    #
    # print("Beginning Training")
    # print(wandb.config)
    # algorithm.train(wandb.config['epochs'])
    # algorithm.save_checkpoint(checkpoint_dir)
