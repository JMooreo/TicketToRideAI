import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)

    def act(self, observation: np.ndarray, action_mask: np.ndarray):
        obs_as_tensor = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(obs_as_tensor.unsqueeze(0)).detach().numpy()
        masked = q_values * action_mask
        filtered = np.where(masked == 0, -np.inf, masked)
        return int(np.argmax(filtered, axis=1))
