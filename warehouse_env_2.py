# warehouse_env_fixed.py

import gymnasium as gym
import numpy as np

class WarehouseEnv(gym.Env):
    """
    Fixed Custom Environment for a warehouse robot that follows the gymnasium.Env interface.
    Key improvements:
    - No immediate termination on obstacles (agent bounces back)
    - Distance-based reward shaping
    - Maximum episode length to prevent infinite loops
    - Better obstacle placement
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, grid_size=10, max_steps=200):
        super(WarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 4 discrete actions (Up, Down, Left, Right)
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation Space: The entire grid flattened into a 1D array.
        # Values: 0=empty, 1=agent, 2=obstacle, 3=target
        self.observation_space = gym.spaces.Box(
            low=0, high=3, shape=(grid_size * grid_size,), dtype=np.int32
        )

        self._agent_location = None
        self._target_location = None
        self._obstacle_locations = []

    def _place_randomly(self):
        """Helper function to place the agent, target, and obstacles randomly."""
        # Generate more obstacles for a more interesting environment
        num_obstacles = min(5, self.grid_size // 2)  # Scale with grid size
        
        locations = set()
        # Keep trying until we have enough unique locations
        while len(locations) < 2 + num_obstacles:
            locations.add(tuple(np.random.randint(0, self.grid_size, size=2)))
        
        loc_list = list(locations)
        self._agent_location = np.array(loc_list[0])
        self._target_location = np.array(loc_list[1])
        self._obstacle_locations = [np.array(loc_list[i]) for i in range(2, 2 + num_obstacles)]

    def _get_observation(self):
        """Generates the state representation as a flattened grid."""
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for obs_loc in self._obstacle_locations:
            obs[obs_loc[0], obs_loc[1]] = 2  # Obstacle
        obs[self._target_location[0], self._target_location[1]] = 3  # Target
        obs[self._agent_location[0], self._agent_location[1]] = 1  # Agent
        return obs.flatten()

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # In warehouse_env_fixed.py

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
    
        # --- ADDED LINE ---
        # This is the crucial fix: It seeds numpy's random number generator.
        if seed is not None:
            np.random.seed(seed)

        self._place_randomly()
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1
        previous_location = self._agent_location.copy()
        
        # Calculate distance to target before move (for reward shaping)
        prev_distance = self._manhattan_distance(self._agent_location, self._target_location)

        # Apply action
        if action == 0:  # Up
            new_location = self._agent_location + [-1, 0]
        elif action == 1:  # Down
            new_location = self._agent_location + [1, 0]
        elif action == 2:  # Left
            new_location = self._agent_location + [0, -1]
        elif action == 3:  # Right
            new_location = self._agent_location + [0, 1]
        
        terminated = False
        reward = 0
        
        # Check if new position is valid
        valid_move = True
        
        # Check for out-of-bounds moves
        if not (0 <= new_location[0] < self.grid_size and \
                0 <= new_location[1] < self.grid_size):
            valid_move = False
            reward = -10  # Penalty for trying to go out of bounds
        
        # Check for hitting an obstacle
        else:
            for obs_loc in self._obstacle_locations:
                if np.array_equal(new_location, obs_loc):
                    valid_move = False
                    reward = -10  # Penalty for hitting obstacle
                    break
        
        # If move is valid, update position
        if valid_move:
            self._agent_location = new_location
            
            # Check if reached target
            if np.array_equal(self._agent_location, self._target_location):
                reward = 100
                terminated = True
            else:
                # Distance-based reward shaping
                new_distance = self._manhattan_distance(self._agent_location, self._target_location)
                distance_reward = (prev_distance - new_distance) * 1.0  # Reward for getting closer
                reward = distance_reward - 0.1  # Small step penalty
        
        # Check for episode timeout
        if self.current_step >= self.max_steps:
            terminated = True
            reward -= 50  # Penalty for not completing in time
        
        observation = self._get_observation()
        truncated = False
        
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Renders the environment for visualization."""
        grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
        for obs_loc in self._obstacle_locations:
            grid[obs_loc[0], obs_loc[1]] = 'X'
        grid[self._target_location[0], self._target_location[1]] = 'T'
        grid[self._agent_location[0], self._agent_location[1]] = 'A'
        render_str = "\r" + "\n".join(" ".join(row) for row in grid) + "\n"
        if mode == 'human':
            print(render_str)
        return render_str

    def close(self):
        pass