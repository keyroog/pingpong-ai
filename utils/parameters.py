# Actions for Pong
_STAY = 0
_MOVE_UP = 1
_MOVE_DOWN = 2
_ACTIONS = [_STAY, _MOVE_UP, _MOVE_DOWN]

# Grid partitions for state discretization
_GRID_PARTITIONS = 12  # Number of bins for discretization

#train episodes: 100 - test episodes: 100
# Parametri per il Q
# Q_Parameters = {
#     "epsilon_start": 0.9,     # Mantieni esplorazione iniziale alta
#     "epsilon_end": 0.05,      # Mantieni minima esplorazione finale
#     "alpha": 0.25,            # Learning rate iniziale
#     "alpha_end": 0.05,        # Learning rate finale
#     "gamma": 0.99,            # Discount factor
#     "epsilon_decay": 150000,  # Esplorazione diminuisce più lentamente
#     "alpha_decay": 0.999      # Decadimento learning rate
# }
# SARSA parameters
SARSA_Parameters = {
    "epsilon_start": 0.9,     # Mantieni esplorazione iniziale alta
    "epsilon_end": 0.05,      # Mantieni minima esplorazione finale
    "alpha": 0.25,            # Learning rate iniziale
    "alpha_end": 0.05,        # Learning rate finale
    "gamma": 0.99,            # Discount factor
    "epsilon_decay": 150000,  # Esplorazione diminuisce più lentamente
    "alpha_decay": 0.999      # Decadimento learning rate
}

Q_Parameters = {
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "alpha": 0.2,             # Learning rate leggermente più basso
    "alpha_end": 0.05,
    "gamma": 0.99,
    "epsilon_decay": 300000,  # Esplorazione dura più a lungo
    "alpha_decay": 0.9995     # Decadimento più lento
}

# Reward values for Pong
REWARD_Values = {
    "hit": 1.0,               # Reward for hitting the ball
    "miss": -1.0,            # Penalty for missing the ball
    "default": 0.0,           # Default reward for non-terminal states
}

# Expose actions and states
_ACTIONS = [_STAY, _MOVE_UP, _MOVE_DOWN]