from Environments.TTREnv import TTREnv
from actors.Agent import Agent


class RandomAgent(Agent):
    def act(self, env: TTREnv) -> int:
        return env.action_space.sample()
