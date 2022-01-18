import unittest

import gym

from src.DeepQLearning.Agent import Agent
from src.Environments.TTREnv import TTREnv


class DeepQNetworkTest(unittest.TestCase):

    def setUp(self):
        self.agent = Agent.random()
        self.env = TTREnv()

    def test_random_agent(self):
        self.assertIsNotNone(self.agent)

    def test_random_agent_in_ttr_env(self):
        done = False
        observation = self.env.reset()
        self.agent.epsilon = 1

        while not done:
            action = self.agent.choose_action_id(observation, self.env.action_space)
            self.assertTrue(self.env.action_space.contains(action))
            observation_, reward, done, info = self.env.step(action)

    def test_nn_agent_in_ttr_env(self):
        done = False
        observation = self.env.reset()
        self.agent.epsilon = 0

        while not done:
            action = self.agent.choose_action_id(observation, self.env.action_space)
            self.assertTrue(self.env.action_space.contains(action))
            observation_, reward, done, info = self.env.step(action)

    def test_random_agent_in_gym_environment(self):
        self.env = gym.make("CartPole-v0")
        done = False
        observation = self.env.reset()
        self.agent.epsilon = 1

        while not done:
            action = self.agent.choose_action_id(observation, self.env.action_space)
            self.assertTrue(self.env.action_space.contains(action))
            observation_, reward, done, info = self.env.step(action)

    def test_nn_agent_in_gym_environment(self):
        self.env = gym.make("CartPole-v0")
        done = False
        observation = self.env.reset()
        self.agent = Agent(gamma=0.99, epsilon=0, lr=0.03, input_dims=(4,), batch_size=32, n_actions=2)

        while not done:
            action = self.agent.choose_action_id(observation, self.env.action_space)
            self.assertTrue(self.env.action_space.contains(action))
            observation_, reward, done, info = self.env.step(action)
