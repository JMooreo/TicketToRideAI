import numpy as np
import torch
from torch import nn


# class Network(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         in_features = int(np.prod(env.observation_space.shape))
#
#         self.net = nn.Sequential(
#             nn.Linear(in_features, 64),
#             nn.Tanh(),
#             nn.Linear(64, env.action_space.n))
#
#     def forward(self, x):
#         return self.net(x)
#
#     def act(self, observation: np.ndarray, action_mask: np.ndarray):
#         obs_as_tensor = torch.as_tensor(observation, dtype=torch.float32)
#         q_values = self(obs_as_tensor.unsqueeze(0)).detach().numpy()
#         masked = q_values * action_mask
#         filtered = np.where(masked == 0, -np.inf, masked)
#         return int(np.argmax(filtered, axis=1))

    # def act(self, players_last_turn: np.ndarray, observation: np.ndarray, action_mask: np.ndarray):
    #     players_last_turn_as_tensor = torch.from_numpy(players_last_turn).unsqueeze(0).unsqueeze(0)
    #     # print(players_last_turn_as_tensor)
    #     observation_as_tensor = torch.from_numpy(observation).unsqueeze(0)
    #     activations = self.forward(players_last_turn_as_tensor, observation_as_tensor)
    #     # print("ACTIVATIONS")
    #     # print(activations)
    #     q_values = activations.detach().numpy()
    #     # print("Q Values")
    #     # print(q_values)
    #     masked = q_values * action_mask
    #     # print("MASKED")
    #     # print(masked)
    #     filtered = np.where(masked == 0, -np.inf, masked)
    #     # print("FILTERED")
    #     # print(filtered)
    #     action = int(np.argmax(filtered, axis=1))
    #     # print("Action", action)
    #     return action

