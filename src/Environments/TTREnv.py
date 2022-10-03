from src.actions.PassAction import PassAction
from src.game.Game import Game
from src.game.Map import USMap
from src.game.Player import Player
from src.game.enums.GameState import GameState
from src.training.ActionSpace import ActionSpace
from src.training.GameTree import GameTree
from src.training.ObservationSpace import ObservationSpace


class TTREnv:
    def __init__(self, tree=None, action_space=None, observation_space=None):
        self.tree = tree if tree else GameTree(Game(players=2, game_map=USMap()))
        self.action_space = action_space if action_space else ActionSpace(self.tree.game)
        self.observation_space = observation_space if observation_space else ObservationSpace(self.tree.game)

    def step(self, action_id: int):
        training_player = self.tree.game.current_player()
        previous_points = training_player.points_from_routes() + \
                          training_player.points_from_destinations()

        if action_id < 0:
            action = PassAction(self.tree.game)
        else:
            action = self.action_space.get_action_by_id(action_id)

        self.tree.next(action)

        observation = self.observation_space.to_np_array()
        reward = training_player.points_from_routes() + \
                 training_player.points_from_destinations() - previous_points

        done = self.tree.game.state == GameState.GAME_OVER
        info = {}

        return observation, reward, done, info

    def reset(self):
        game = Game.us_game()
        self.tree = GameTree(game)
        self.action_space = ActionSpace(game)
        self.observation_space = ObservationSpace(game)

        return self.observation_space.to_np_array()

    def render(self):
        return print(self.tree.game)
#
#
# def replay_buffer_filled_random(env, min_replay_size, buffer_size):
#     replay_buffer = deque(maxlen=buffer_size)
#
#     observation = env.reset()
#
#     for _ in range(min_replay_size):
#         action = env.action_space.sample()
#         new_observation, reward, done, _ = env.step(action)
#         transition = (observation, action, reward, done, new_observation)
#         replay_buffer.append(transition)
#         observation = new_observation
#
#         if done:
#             observation = env.reset()
#
#     return replay_buffer

#
# def main():
#     checkpoint_directory = "D:/Programming/TicketToRideMCCFR_TDD/checkpoints"
#     env = TTREnv()


#
#     GAMMA = 0.99
#     BATCH_SIZE = 32
#     BUFFER_SIZE = 50000
#     MIN_REPLAY_SIZE = 1000
#     EPSILON_START = 1.0
#     EPSILON_END = 0.02
#     EPSILON_DECAY = 50000
#     TARGET_UPDATE_FREQ = 1000
#     LEARNING_RATE = 5e-3
#
#     replay_buffer = replay_buffer_filled_random(env, MIN_REPLAY_SIZE, BUFFER_SIZE)
#     reward_buffer = deque([0.0], maxlen=100)
#
#     episode_reward = 0.0
#
#     online_net = load_latest_checkpoint(checkpoint_directory)
#     target_net = Network(env)
#
#     target_net.load_state_dict(online_net.state_dict())
#
#     optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
#     observation = env.reset()
#
#     checkpoint_scores = []
#     scores, eps_history = [], []
#     highest_checkpoint_score = -1000
#     done = False
#
#     for step in itertools.count():
#         epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
#         random_sample = random.random()
#
#         # Handle the case where the player can't do anything.
#         if sum(env.action_space.valid_action_mask()) == 0:
#             print(env.tree.current_node)
#             print(env.tree.game)
#             env.tree.current_node = env.tree.current_node.pass_turn()
#             continue
#
#         if not done:
#             if random_sample <= epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = online_net.act(observation, env.action_space.valid_action_mask())
#
#             new_observation, reward, done, _ = env.step(action)
#             transition = (observation, action, reward, done, new_observation)
#             replay_buffer.append(transition)
#             observation = new_observation
#
#             episode_reward += reward
#         else:
#             observation = env.reset()
#             reward_buffer.append(episode_reward)
#             eps_history.append(epsilon)
#             scores.append(episode_reward)
#             checkpoint_scores.append(episode_reward)
#             highest_checkpoint_score = max(highest_checkpoint_score, episode_reward)
#             episode_reward = 0.0
#
#         transitions = random.sample(replay_buffer, BATCH_SIZE)
#
#         observation_tensor = torch.as_tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
#         action_tensor = torch.as_tensor(np.array([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
#         reward_tensor = torch.as_tensor(np.array([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
#         done_tensor = torch.as_tensor(np.array([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
#         new_observation_tensor = torch.as_tensor(np.array([t[4] for t in transitions]), dtype=torch.float32)
#
#         target_q_values = target_net(new_observation_tensor)
#         max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
#
#         targets = reward_tensor + GAMMA * (1 - done_tensor) * max_target_q_values
#
#         q_values = online_net(observation_tensor)
#         action_q_values = torch.gather(input=q_values, dim=1, index=action_tensor)
#
#         loss = nn.MSELoss().forward(action_q_values, targets)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % TARGET_UPDATE_FREQ == 0:
#             target_net.load_state_dict(online_net.state_dict())
#
#             avg_score = np.mean(scores[-100:])
#
#             print(f'step {step}   '
#                   f'average score {round(avg_score, 2)}   '
#                   f'epsilon {round(epsilon, 2)}   '
#                   f'learning rate {LEARNING_RATE}')
#
#         if step % 10000 == 0 and step != 0:
#             file_path = f"{checkpoint_directory}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}" \
#                         f"game_{len(scores)}_avg_{np.mean(checkpoint_scores)}.pkl"
#             save_checkpoint(online_net, file_path)
#             highest_checkpoint_score = -1000
#             checkpoint_scores = []
#
#         if step > 100001:
#             x = np.arange(len(scores))
#             filename = '../DeepQLearning/results.png'
#             plotLearning(x, np.array(scores), eps_history, filename)
#             break
#
#
# if __name__ == '__main__':
#     main()
