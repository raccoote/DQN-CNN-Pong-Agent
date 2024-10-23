

# dependencies

# pip install torch
# pip install gymnasium[classic_control]
# pip install gymnasium[atari]
# pip install gymnasium[accept-rom-license]

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle


# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 1000
TAU = 0.01
LR = 0.0001
REPLAY_MEMORY = 100000

render = False
num_episodes = 30






if render:
    env = gym.make("ALE/Pong-v5", render_mode='human')
else:
    env = gym.make("ALE/Pong-v5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN CNN class
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # 3 conv layers
        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


        conv_output_size = self.conv_output_dim()
        lin1_output_size = 512

        # 2 fully connected layers
        self.l2 = nn.Sequential(
            nn.Linear(conv_output_size, lin1_output_size),
            nn.ReLU(),
            nn.Linear(lin1_output_size, output_dim)
        )


    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self.l1(x)
        x = x.reshape(x.size(0), -1)  # flatten
        actions = self.l2(x)
        return actions




output_dim = env.action_space.n
input_dim = (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(REPLAY_MEMORY)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def save_model(policy_net, target_net, optimizer, filename="model.pth"):
    state = {
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, filename)

def load_model(policy_net, target_net, optimizer, filename="model.pth"):
    state = torch.load(filename)
    policy_net.load_state_dict(state['policy_net_state_dict'])
    target_net.load_state_dict(state['target_net_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    print(f"Model loaded from {filename}")

def save_graph(durations, filename="episode_durations.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(durations, f)

def load_graph(filename):
    with open(filename, 'rb') as f:
            return pickle.load(f)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


episode_durations = []
average_Q_values = []
average_rewards = []

#load_model(policy_net, target_net, optimizer, filename="model.pth")
#episode_durations = load_graph(filename="episode_durations.pkl")
#average_Q_values = load_graph(filename="average_Q_values.pkl")
#average_rewards = load_graph(filename="average_rewards.pkl")

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    state = state.permute(0, 3, 1, 2)

    total_reward = 0
    total_Q_value = 0
    steps = 0

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_reward += reward.item()
        done = terminated or truncated

        if render:
            env.render()

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = next_state.permute(0, 3, 1, 2)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        if state is not None:
            q_value = policy_net(state).max(1)[0].item()
            total_Q_value += q_value
            steps += 1

        # Soft update of the target net
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            average_Q_values.append(total_Q_value / steps)
            average_rewards.append(total_reward / steps)
            print(t+1)
            break
    save_model(policy_net, target_net, optimizer, filename="model.pth")
    save_graph(episode_durations, filename="episode_durations.pkl")
    save_graph(average_Q_values, filename="average_Q_values.pkl")
    save_graph(average_rewards, filename="average_rewards.pkl")
    torch.cuda.empty_cache()



# results
plt.figure(figsize=(10,5))
plt.subplot(3, 1, 1)
plt.plot(episode_durations)
plt.title('Episode durations')
plt.ylabel('Duration')

plt.subplot(3, 1, 2)
plt.plot(average_Q_values)
plt.title('Average Q-values per episode')
plt.ylabel('Average Q-value')

plt.subplot(3, 1, 3)
plt.plot(average_rewards)
plt.title('Average rewards per episode')
plt.ylabel('Average Reward')
plt.xlabel('Episode')
plt.legend()

plt.tight_layout()
plt.show()
