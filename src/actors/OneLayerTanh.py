import numpy as np
from torch import nn

from src.actors.NeuralNet import NeuralNet


class OneLayerTanh(NeuralNet):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))
