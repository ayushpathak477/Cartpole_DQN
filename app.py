# app.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from flask import Flask, request, jsonify, render_template
import numpy as np

# Import the custom environment
from warehouse_env_fixed import WarehouseEnv 

# --- Actor-Critic Network Definition ---
# This must match the architecture of the saved model
class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        shared_layers = nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(shared_layers, nn.Linear(256, n_actions))
        self.critic = nn.Sequential(shared_layers, nn.Linear(256, 1))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# --- Global Variables & Model Loading ---
device = torch.device("cpu")
env = WarehouseEnv(grid_size=8, max_steps=100)
n_observations = np.prod(env.observation_space.shape)
n_actions = env.action_space.n

# Make sure this points to your best trained model
MODEL_PATH = "warehouse_ppo_solved.pth" 
agent = ActorCritic(n_observations, n_actions).to(device)
agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
agent.eval()  # Set the agent to evaluation mode

print(f"Model {MODEL_PATH} loaded successfully on {device}.")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- API Endpoint Definitions ---

# Endpoint to serve the main HTML page
@app.route("/")
def index():
    # Renders the index.html file from the 'templates' folder
    return render_template('index.html')

# Endpoint to reset the environment with a random layout
@app.route("/reset", methods=['POST'])
def reset():
    state, _ = env.reset()
    return jsonify({'state': state.tolist()})

# Endpoint to reset with a pre-determined "good" seed
# In app.py

@app.route("/reset_good_seed", methods=['POST'])
def reset_good_seed():
    # This seed will now only affect the environment's layout
    good_seed = 42
    
    # Manually seed the environment's action space and reset
    # This keeps the layout the same but does NOT affect torch's randomness
    env.action_space.seed(good_seed)
    state, _ = env.reset(seed=good_seed)
    
    return jsonify({'state': state.tolist()})

# Endpoint to take a step in the environment
@app.route("/step", methods=['POST'])
def step():
    state_list = request.json['state']
    state_tensor = torch.Tensor(state_list).to(device)
    
    # Get agent's action by sampling from its policy
    with torch.no_grad():
        logits = agent.actor(state_tensor.unsqueeze(0))
        probs = Categorical(logits=logits)
        action = probs.sample().item() # Sample, don't use argmax
    
    # Perform the step in the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    return jsonify({
        'next_state': next_state.tolist(),
        'reward': reward,
        'done': terminated or truncated
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)