import itertools
import random
from collections import deque

import gym
import numpy as np
import torch
from torch import nn

if __name__ == "__main__":
    GAMMA = 0.99
    BATCH_SIZE = 32
    BUFFER_SIZE = 50000
    MIN_REPLAY_SIZE = 1000
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 10000
    TARGET_UPDATE_FREQ = 1000

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

        def act(self, observation):
            obs_as_tensor = torch.as_tensor(observation, dtype=torch.float32)
            q_values = self(obs_as_tensor.unsqueeze(0))

            max_q_index = torch.argmax(q_values, dim=1)[0]
            return max_q_index.detach().item()

    env = gym.make("CartPole-v0")

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    reward_buffer = deque([0.0], maxlen=100)

    episode_reward = 0.0

    online_net = Network(env)
    target_net = Network(env)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)
    observation = env.reset()

    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        new_observation, reward, done, _ = env.step(action)
        transition = (observation, action, reward, done, new_observation)
        replay_buffer.append(transition)
        observation = new_observation

        if done:
            observation = env.reset()

    observation = env.reset()

    for step in itertools.count():
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(observation)

        new_observation, reward, done, _ = env.step(action)
        transition = (observation, action, reward, done, new_observation)
        replay_buffer.append(transition)
        observation = new_observation

        episode_reward += reward

        if step > 80000:
            env.render()

        if done:
            observation = env.reset()
            reward_buffer.append(episode_reward)
            episode_reward = 0.0

        transitions = random.sample(replay_buffer, BATCH_SIZE)

        observation_tensor = torch.as_tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
        action_tensor = torch.as_tensor(np.array([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
        reward_tensor = torch.as_tensor(np.array([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
        done_tensor = torch.as_tensor(np.array([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
        new_observation_tensor = torch.as_tensor(np.array([t[4] for t in transitions]), dtype=torch.float32)

        target_q_values = target_net(new_observation_tensor)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = reward_tensor + GAMMA * (1 - done_tensor) * max_target_q_values

        q_values = online_net(observation_tensor)
        action_q_values = torch.gather(input=q_values, dim=1, index=action_tensor)

        loss = nn.HuberLoss().forward(action_q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        if step % 1000 == 0:
            print()
            print("STEP", step)
            print("AVERAGE REWARD", np.mean(reward_buffer))
