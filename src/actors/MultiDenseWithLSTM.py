import numpy as np
import torch
from torch import nn


class MultiDenseWithLSTM(nn.Module):
    def __init__(self, env):
        super().__init__()

        action_size = env.action_space.n
        observation_size = int(np.prod(env.observation_space.shape))

        self.lstm = nn.LSTM(input_size=action_size, hidden_size=128, num_layers=3, batch_first=True)

        self.dense1 = nn.Linear(observation_size + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, action_size)

    def forward(self, lstm_input, environment_observation):
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]

        x = torch.cat([lstm_out, environment_observation], dim=-1)

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)

        return x
