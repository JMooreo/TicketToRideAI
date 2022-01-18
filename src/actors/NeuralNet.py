import torch
from torch import nn


class NeuralNet(nn.Module):
    def forward(self, x):
        return self.net(x)

    def act(self, observation):
        obs_as_tensor = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(obs_as_tensor.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        return max_q_index.detach().item()
