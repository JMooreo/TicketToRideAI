import numpy as np
import torch
from torch import nn

from src.actors.MultiLayerReLU import MultiLayerReLU


class TTRValidator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.actor = MultiLayerReLU(env)

    def forward(self, x):
        return self.actor.forward(x)

    def act(self, observation):
        obs_as_tensor = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(obs_as_tensor.unsqueeze(0)).detach().numpy()
        action_mask = self.env.action_space.valid_action_mask()

        if sum(action_mask) == 0:
            return -1  # TTR Env will interpret this as a pass since no action can be taken.

        masked = q_values * action_mask
        filtered = np.where(masked == 0, -np.inf, masked)

        return int(np.argmax(filtered, axis=1))
