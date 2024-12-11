from agents.base_agent import BaseAgent
from utils.parameters import SARSA_Parameters, _ACTIONS


class SARSAAgent(BaseAgent):
    """
    SARSA implementation.
    """
    def __init__(self):
        super().__init__(
            actions=_ACTIONS,
            epsilon_start=SARSA_Parameters["epsilon_start"],
            epsilon_end=SARSA_Parameters["epsilon_end"],
            epsilon_decay=SARSA_Parameters["epsilon_decay"],
            alpha=SARSA_Parameters["alpha"],
            alpha_end=SARSA_Parameters["alpha_end"],
            alpha_decay=SARSA_Parameters["alpha_decay"],
            gamma=SARSA_Parameters["gamma"],
        )

    def observe(self, state, action, reward, next_state, next_action):
        """
        Update the Q-value using the SARSA equation.
        """
        old_q = self.q_table[(state, action)]
        next_q = self.q_table[(next_state, next_action)]
        new_q = old_q + self.alpha * (reward + self.gamma * next_q - old_q)
        self.q_table[(state, action)] = new_q

        # Update learning rate
        self.update_learning_rate()