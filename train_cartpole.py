# train_cartpole_final.py

import gymnasium as gym
import math
import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Hyperparameters: Balanced for Stability and Performance ---
LEARNING_RATE = 1e-4          # A stable learning rate that showed good results.
BATCH_SIZE = 128              # A standard batch size for stable updates.
GAMMA = 0.99                  # Standard discount factor.
EPS_START = 0.9               # Start with high exploration.
EPS_END = 0.05                # Keep a small amount of exploration.
EPS_DECAY = 10000             # Slow decay to allow for thorough exploration.
TARGET_UPDATE_FREQUENCY = 500 # Step-based update for stability.
MEMORY_SIZE = 10000           # A reasonably sized replay buffer.
NUM_EPISODES = 1500           # Increased episodes for sufficient training time.

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Experience Replay Buffer ---
Experience = collections.namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """A simple circular buffer for storing experiences."""
    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)

# --- 2. DQN Architecture ---
class DQN(nn.Module):
    """
    DQN with larger layers for more capacity, which proved effective.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        """Defines the forward pass of the network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- 3. The DQN Agent ---
class DQNAgent:
    def __init__(self, n_observations, n_actions):
        self.n_actions = n_actions
        self.steps_done = 0

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of optimization using Double DQN."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN implementation for stable Q-value estimation
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

# --- 4. Main Training Loop ---
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(n_observations, n_actions)

    episode_durations = []
    total_steps = 0

    print("Starting training...")
    print(f"Using device: {device}")

    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        t = 0
        while not done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            agent.optimize_model()

            total_steps += 1
            t += 1
            
            # Update the target network based on total steps for stability
            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        episode_durations.append(t)
        
        if (i_episode + 1) % 50 == 0:
            avg_duration = np.mean(episode_durations[-100:])
            print(f"Episode {i_episode+1}/{NUM_EPISODES} | Average Duration (last 100): {avg_duration:.2f}")

        # Check against the correct solved condition for CartPole-v1
        if len(episode_durations) >= 100:
            if np.mean(episode_durations[-100:]) >= 475.0:
                print(f"\nEnvironment solved in {i_episode+1} episodes!")
                torch.save(agent.policy_net.state_dict(), "cartpole_dqn_solved.pth")
                break
                
    print('Complete')
    env.close()