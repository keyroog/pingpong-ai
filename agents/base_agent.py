import random
import math
from collections import defaultdict
import pickle


class BaseAgent:
    def __init__(self, actions, **kwargs):
        """
        Initialize the base agent with dynamic parameters.
        :param actions: List of available actions.
        :param kwargs: Dynamic parameters such as epsilon, alpha, etc.
        """
        self.actions = actions

        # Default values
        self.epsilon_start = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.1)
        self.epsilon_decay = kwargs.get("epsilon_decay", 200)
        self.alpha = kwargs.get("alpha", 0.1)
        self.alpha_end = kwargs.get("alpha_end", 0.01)
        self.alpha_decay = kwargs.get("alpha_decay", 0.99)
        self.gamma = kwargs.get("gamma", 0.99)

        self.steps_done = 0
        self.visit_count = defaultdict(int)  # Visit count for Q-learning
        self.q_table = defaultdict(float)  # Default Q-table

    def update_parameters(self, **kwargs):
        """
        Update agent parameters dynamically.
        :param kwargs: Parameters to update (e.g., epsilon, alpha, etc.).
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


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

    def save(self, filepath):
        """
        Save the Q-table and agent parameters to a file.
        :param filepath: Path to the file where the Q-table will be saved.
        """
        data = {
            "q_table": dict(self.q_table),  # Convert defaultdict to regular dict
            "parameters": {
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "alpha": self.alpha,
                "alpha_end": self.alpha_end,
                "alpha_decay": self.alpha_decay,
                "gamma": self.gamma,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """
        Load the Q-table and agent parameters from a file.
        :param filepath: Path to the file where the Q-table is stored.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data["q_table"])
        for key, value in data["parameters"].items():
            if hasattr(self, key):
                setattr(self, key, value)