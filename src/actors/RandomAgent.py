from Environments.TTREnv import TTREnv
from actors.Agent import Agent


class RandomAgent(Agent):
    def act(self, env: TTREnv):
        return env.action_space.sample()
