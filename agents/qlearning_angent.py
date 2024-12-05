from agents.base_agent import BaseAgent
from utils.parameters import Q_Parameters, _ACTIONS


class QLearningAgent(BaseAgent):
    """
    Q-Learning implementation with visit count-based learning rate adjustment.
    """
    def __init__(self):
        super().__init__(
            actions=_ACTIONS,
            epsilon_start=Q_Parameters["epsilon_start"],
            epsilon_end=Q_Parameters["epsilon_end"],
            epsilon_decay=Q_Parameters["epsilon_decay"],
            alpha=Q_Parameters["alpha"],
            alpha_end=Q_Parameters["alpha_end"],
            alpha_decay=Q_Parameters["alpha_decay"],
            gamma=Q_Parameters["gamma"],
        )

    def observe(self, state, action, reward, next_state):
        """
        Update the Q-value using the Bellman equation and adjust learning rate based on visit count.
        """
        # Increment visit count
        self.visit_count[(state, action)] += 1

        # Dynamically adjust learning rate based on visit count
        visit_count = self.visit_count[(state, action)]
        adjusted_alpha = max(self.alpha_end, self.alpha / (1 + visit_count))  # Decay alpha as visits increase

        # Update Q-value using the adjusted learning rate
        old_q = self.q_table[(state, action)]
        next_max = max([self.q_table[(next_state, a)] for a in self.actions])
        new_q = old_q + adjusted_alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[(state, action)] = new_q

        # Update epsilon or any other time-dependent parameters
        self.update_learning_rate()