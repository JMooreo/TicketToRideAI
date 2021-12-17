import math

from gym import Env
from gym.spaces import Discrete

from src.DeepQLearning.DeepQNetwork import Agent
import numpy as np

from src.DeepQLearning.utils import plotLearning
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.training.ActionSpace import ActionSpace


class GameEnv(Env):
    def __init__(self):
        self.game = Game([Player(), Player()], USMap())
        self.action_space = Discrete(len(ActionSpace(self.game)))
        self.observation_space = Discrete

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass


def do_episode(agent, env):
    score = 0
    done = False
    observation = env.reset()

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

    return score


def main():
    env = GameEnv()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.005)
    scores, eps_history = [], []
    n_games = 100

    for i in range(n_games):
        score = do_episode(agent, env)
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f'episode {i} score {round(score, 2)} '
              f'average score {round(avg_score, 2)} '
              f'epsilon {round(agent.epsilon, 2)} '
              f' learning rate {agent.lr}')

        agent.lr = max(agent.lr - 5e-5, 1e-3)

    x = np.array([i + 1 for i in range(n_games)])
    filename = 'lunar_lander.png'
    plotLearning(x, np.array(scores), eps_history, filename)


# def find_learning_rate():
#     minp = 0
#     maxp = 100
#
#     minv = math.log(1e-7)
#     maxv = math.log(1e-1)
#
#     scale = (maxv-minv) / (maxp-minp)
#
#     rates = [np.exp(minv + scale*(position-minp)) for position in range(100)]
#     print(rates)
#
#     losses = []
#     env = gym.make('LunarLander-v2')
#
#     for idx, rate in enumerate(rates):
#         agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
#                       eps_end=0.01, input_dims=[8], lr=rate)
#         for i in range(10):
#             print(f"Doing Episode {(idx+1)*i + i}/{len(rates) * 10}")
#             do_episode(agent, env)
#
#         loss = agent.calculate_loss().item()
#         losses.append(min(1000, loss))
#
#     print(rates)
#     print(losses)
#
#     import matplotlib.pyplot as plt
#
#     plt.plot(rates, losses)
#     plt.title('Learning Rate vs Loss')
#     plt.xlabel('Learning Rate')
#     plt.ylabel('Losses')
#     plt.show()


if __name__ == '__main__':
   main()
