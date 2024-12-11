import random
import math
from collections import defaultdict


class BaseAgent:
    """
    Base class for RL agents.
    """
    def __init__(self, actions, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=200,
                 alpha=0.1, alpha_end=0.01, alpha_decay=0.99, gamma=0.99):
        """
        :param actions: List of available actions.
        :param epsilon_start: Initial epsilon for exploration.
        :param epsilon_end: Minimum epsilon.
        :param epsilon_decay: Decay rate for epsilon.
        :param alpha: Initial learning rate.
        :param alpha_end: Minimum learning rate.
        :param alpha_decay: Decay factor for learning rate.
        :param gamma: Discount factor for future rewards.
        """
        self.actions = actions
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.gamma = gamma

        self.steps_done = 0
        self.visit_count = defaultdict(int)  # Visit count for Q-learning
        self.q_table = defaultdict(float)  # Default Q-table

    def get_action(self, state):
        """
        Chooses an action based on epsilon-greedy policy.
        :param state: Current state.
        :return: Action selected.
        """
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        threshold = random.uniform(0, 1)

        if threshold < eps_threshold:
            return random.choice(self.actions)  # Explore
        else:
            q_values = [self.q_table[(state, action)] for action in self.actions]
            max_q = max(q_values)
            return self.actions[q_values.index(max_q)]  # Exploit

    def get_best_action(self, state):
        """
        Chooses the best action based on the current Q-values.
        :param state: Current state.
        :return: Best action.
        """
        q_values = [self.q_table[(state, action)] for action in self.actions]
        max_q = max(q_values)
        return self.actions[q_values.index(max_q)]

    def observe(self, state, action, reward, next_state):
        """
        Updates the Q-value. To be implemented by subclasses.
        """
        raise NotImplementedError("The 'observe' method must be implemented in a subclass.")

    def update_learning_rate(self):
        """
        Decays the learning rate over time.
        """
        self.alpha = max(self.alpha_end, self.alpha * self.alpha_decay)