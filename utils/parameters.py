# Actions for Pong
_STAY = 0
_MOVE_UP = 1
_MOVE_DOWN = 2
_ACTIONS = [_STAY, _MOVE_UP, _MOVE_DOWN]

# Grid partitions for state discretization
_GRID_PARTITIONS = 12  # Number of bins for discretization

#train episodes: 25000 - test episodes: 100
# Parametri per il Q
Q_Parameters = {
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "alpha": 0.25,
    "alpha_end": 0.05,
    "gamma": 0.99,
    "epsilon_decay": 100000,
    "alpha_decay": 0.999
}

# SARSA parameters
SARSA_Parameters = {
    "epsilon_start": 0.9,     # Initial epsilon for exploration
    "epsilon_end": 0.05,      # Minimum epsilon
    "alpha": 0.25,            # Learning rate
    "alpha_end": 0.05,        # Minimum learning rate
    "gamma": 0.99,            # Discount factor for future rewards
    "epsilon_decay": 10000,   # Decay rate for epsilon
    "alpha_decay": 0.999     # Decay rate for learning rate  # Number of partitions for discretization
}

# Reward values for Pong
REWARD_Values = {
    "hit": 1.0,               # Reward for hitting the ball
    "miss": -1.0,            # Penalty for missing the ball
    "default": 0.0,           # Default reward for non-terminal states
}

# Expose actions and states
_ACTIONS = [_STAY, _MOVE_UP, _MOVE_DOWN]