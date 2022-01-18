import unittest

from src.DeepQLearning.Agent import Agent
from src.Environments.TTREnv import TTREnv


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.agent = Agent.random()
        self.env = TTREnv()

    def test_init(self):
        self.assertEqual(0, self.agent.mem_cntr)
        self.assertEqual(self.agent.mem_size, len(self.agent.state_memory))
        self.assertEqual(self.agent.mem_size, len(self.agent.new_state_memory))
        self.assertEqual(self.agent.mem_size, len(self.agent.reward_memory))
        self.assertEqual(self.agent.mem_size, len(self.agent.terminal_memory))

    def test_agent_store_transition(self):
        observation = self.env.reset()
        action = self.agent.choose_action_id(observation, self.env.action_space)
        self.assertTrue(self.env.action_space.contains(action))

        observation_, reward, done, info = self.env.step(action)
        self.agent.store_transition(observation, action, reward, observation_, done)

        self.assertEqual(1, self.agent.mem_cntr)
        self.assertEqual(self.agent.mem_size, len(self.agent.state_memory))
        self.assertEqual(self.agent.mem_size, len(self.agent.new_state_memory))
        self.assertEqual(self.agent.mem_size, len(self.agent.reward_memory))
        self.assertEqual(self.agent.mem_size, len(self.agent.terminal_memory))

    def test_learn_doesnt_decrease_epsilon_until_mem_counter_reached_batch_size(self):
        observation = self.env.reset()

        self.agent.epsilon = 1.0
        while self.agent.mem_cntr < self.agent.batch_size - 1:
            action = self.agent.choose_action_id(observation, self.env.action_space)
            self.assertTrue(self.env.action_space.contains(action))

            observation_, reward, done, info = self.env.step(action)
            self.agent.store_transition(observation, action, reward, observation_, done)
            self.agent.learn()

            self.assertEqual(1.0, self.agent.epsilon)

        action = self.agent.choose_action_id(observation, self.env.action_space)
        self.assertTrue(self.env.action_space.contains(action))

        observation_, reward, done, info = self.env.step(action)
        self.agent.store_transition(observation, action, reward, observation_, done)
        self.agent.learn()

        self.assertGreater(1.0, self.agent.epsilon)
