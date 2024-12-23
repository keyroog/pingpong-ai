# Actions for Pong
_STAY = 0
_MOVE_UP = 1
_MOVE_DOWN = 2
_ACTIONS = [_STAY, _MOVE_UP, _MOVE_DOWN]

# Grid partitions for state discretization
_GRID_PARTITIONS = 12  # Number of bins for discretization

SARSA_Parameters = {
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "alpha": 0.15,            # Learning rate più basso per stabilità
    "alpha_end": 0.02,        # Learning rate finale più basso
    "gamma": 0.99,
    "epsilon_decay": 500000,  # Esplorazione dura molto più a lungo
    "alpha_decay": 0.9999     # Decadimento estremamente lento
}

Q_Parameters = {
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "alpha": 0.15,            # Learning rate più basso per stabilità
    "alpha_end": 0.02,        # Learning rate finale più basso
    "gamma": 0.99,
    "epsilon_decay": 500000,  # Esplorazione dura molto più a lungo
    "alpha_decay": 0.9999     # Decadimento estremamente lento
}

# Reward values for Pong
REWARD_Values = {
    "hit": 1.0,               # Reward for hitting the ball
    "miss": -1.0,            # Penalty for missing the ball
    "default": 0.0,           # Default reward for non-terminal states
}