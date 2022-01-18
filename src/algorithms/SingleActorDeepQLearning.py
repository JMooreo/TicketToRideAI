import os
import pickle
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
from torch import nn

from src.algorithms.algorithm import Algorithm


class SingleActorDeepQLearningAlgorithm(Algorithm):
    def __init__(self, env, network_structure, wandb,
                 gamma=0.99, batch_size=32, buffer_size=50000, min_replay_size=1000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_period=10000, target_update_freq=1000,
                 logging_freq=1000, learning_rate=5e-3, epochs=1):

        # Hyper parameters
        self.GAMMA = gamma
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.MIN_REPLAY_SIZE = min_replay_size
        self.EPSILON = epsilon_start
        self.EPSILON_START = epsilon_start
        self.EPSILON_END = epsilon_end
        self.EPSILON_DECREMENT = (epsilon_start - epsilon_end) / epsilon_decay_period
        self.TARGET_UPDATE_FREQ = target_update_freq
        self.LOGGING_FREQ = logging_freq
        self.wandb = wandb

        # Setup
        self.env = env

        self.online_net = network_structure(env)
        self.target_net = network_structure(env)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        self.replay_buffer = deque(maxlen=buffer_size)
        while len(self.replay_buffer) < self.MIN_REPLAY_SIZE:
            self.do_episode(epsilon=1)

        self.reward_buffer = deque([0.0], maxlen=100)

    def choose_action(self, epsilon, observation):
        random_sample = random.random()
        if random_sample <= epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.online_net.act(observation)

        return action

    def do_episode(self, epsilon):
        observation = self.env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = self.choose_action(epsilon, observation)
            new_observation, reward, done, _ = self.env.step(action)

            if action < 0:
                continue  # Skip this transition because player has to pass. No other option.

            transition = (observation, action, reward, done, new_observation)
            self.replay_buffer.append(transition)

            observation = new_observation
            episode_reward += reward

        self.env.reset()
        return episode_reward

    def train(self, num_episodes):
        self.wandb.watch(self.online_net)

        i = 0
        while i < num_episodes:
            episode_reward = self.do_episode(self.EPSILON)
            self.reward_buffer.append(episode_reward)
            self.EPSILON = max(self.EPSILON - self.EPSILON_DECREMENT, self.EPSILON_END)

            loss = self.learn()

            if i > 0:
                if i % self.TARGET_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                if i % self.LOGGING_FREQ == 0:
                    self.wandb.log({
                        "loss": loss,
                        "episode_reward": episode_reward,
                        "average_reward": np.mean(self.reward_buffer),
                        "epsilon_greedy": self.EPSILON
                    })
            i += 1

    def learn(self):
        transitions = random.sample(self.replay_buffer, self.BATCH_SIZE)

        observation_tensor = torch.as_tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
        action_tensor = torch.as_tensor(np.array([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
        reward_tensor = torch.as_tensor(np.array([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
        done_tensor = torch.as_tensor(np.array([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
        new_observation_tensor = torch.as_tensor(np.array([t[4] for t in transitions]), dtype=torch.float32)

        target_q_values = self.target_net(new_observation_tensor)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = reward_tensor + self.GAMMA * (1 - done_tensor) * max_target_q_values

        q_values = self.online_net(observation_tensor)
        action_q_values = torch.gather(input=q_values, dim=1, index=action_tensor)

        loss = nn.MSELoss().forward(action_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_checkpoint(self, checkpoint_dir):
        file_path = f"{checkpoint_dir}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self.online_net, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_latest_checkpoint(self, checkpoint_dir):
        try:
            file_path = checkpoint_dir + "/" + os.listdir(checkpoint_dir)[-1]
            with open(file_path, "rb") as f:
                agent = pickle.load(f)
                print("\nCHECKPOINT LOADED", file_path)
                print()

            self.online_net = agent
            self.target_net.load_state_dict(self.online_net.state_dict())
        except IndexError:
            print("No checkpoints in directory. Starting from random.")
