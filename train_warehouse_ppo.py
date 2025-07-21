# train_warehouse_ppo.py

import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Import your custom environment
from warehouse_env_fixed import WarehouseEnv 

# --- PPO Hyperparameters ---
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95        # Generalized Advantage Estimation lambda
PPO_CLIP = 0.2           # PPO clipping parameter
PPO_EPOCHS = 10          # Number of optimization epochs per batch
NUM_MINI_BATCHES = 4     # Number of mini-batches to split a batch into
UPDATE_TIMESTEPS = 2048  # Number of steps to collect before updating
NUM_EPISODES = 5000      # Total number of episodes to run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        # Shared network layers
        shared_layers = nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Actor head: learns the policy (what action to take)
        self.actor = nn.Sequential(
            shared_layers,
            nn.Linear(256, n_actions)
        )
        
        # Critic head: learns the value of a state
        self.critic = nn.Sequential(
            shared_layers,
            nn.Linear(256, 1)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# --- Main PPO Training Script ---
if __name__ == "__main__":
    # Initialize Environment
    env = WarehouseEnv(grid_size=8, max_steps=100)
    
    n_observations = np.prod(env.observation_space.shape)
    n_actions = env.action_space.n
    
    agent = ActorCritic(n_observations, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # PPO Storage setup
    states = torch.zeros((UPDATE_TIMESTEPS, n_observations)).to(device)
    actions = torch.zeros(UPDATE_TIMESTEPS).to(device)
    log_probs = torch.zeros(UPDATE_TIMESTEPS).to(device)
    rewards = torch.zeros(UPDATE_TIMESTEPS).to(device)
    dones = torch.zeros(UPDATE_TIMESTEPS).to(device)
    values = torch.zeros(UPDATE_TIMESTEPS).to(device)

    # Training Loop
    global_step = 0
    start_time = time.time()
    
    # Initial state
    next_state, _ = env.reset()
    next_state = torch.Tensor(next_state).to(device)
    next_done = torch.zeros(1).to(device)

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print("Starting training with PPO...")
    print(f"Using device: {device}")
    
    for i_episode in range(NUM_EPISODES):
        episode_reward = 0
        episode_length = 0
        
        for step in range(UPDATE_TIMESTEPS):
            global_step += 1
            
            states[step] = next_state
            dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_state.unsqueeze(0))
                values[step] = value.flatten()
            
            actions[step] = action
            log_probs[step] = log_prob

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_state, next_done = torch.Tensor(next_state).to(device), torch.Tensor([done]).to(device)

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if episode_reward > 50: # Success condition
                    success_count += 1
                
                episode_reward = 0
                episode_length = 0
                next_state, _ = env.reset()
                next_state = torch.Tensor(next_state).to(device)
        
        # --- PPO Update Phase ---
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lambda = 0
            # Get value of the last state
            next_value = agent.get_value(next_state.unsqueeze(0)).reshape(1, -1)
            
            # Calculate advantages using GAE
            for t in reversed(range(UPDATE_TIMESTEPS)):
                if t == UPDATE_TIMESTEPS - 1:
                    next_non_terminal = 1.0 - next_done
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_return = values[t + 1]
                
                delta = rewards[t] + GAMMA * next_return * next_non_terminal - values[t]
                advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda
            
            returns = advantages + values

        # Flatten the batch
        b_states = states.reshape((-1, n_observations))
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Optimizing the policy and value network
        batch_size = b_states.shape[0]
        inds = np.arange(batch_size)
        
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, batch_size, batch_size // NUM_MINI_BATCHES):
                end = start + (batch_size // NUM_MINI_BATCHES)
                mb_inds = inds[start:end]

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_states[mb_inds], b_actions.long()[mb_inds]
                )
                
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                # Policy loss (clipped)
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                loss = pg_loss - 0.01 * entropy.mean() + v_loss * 0.5
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # --- Logging ---
        if (i_episode + 1) % 10 == 0 and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            # Calculate success rate over the last 100 episodes
            recent_success_rate = np.mean([1 if r > 50 else 0 for r in episode_rewards[-100:]])
            
            print(f"Episode {i_episode+1}/{NUM_EPISODES}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {recent_success_rate:.2%}\n")

    print('Training complete!')
    torch.save(agent.state_dict(), "warehouse_ppo_solved.pth")
    env.close()