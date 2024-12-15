from agents.base_agent import BaseAgent
from utils.parameters import SARSA_Parameters, _ACTIONS

class SARSAAgent(BaseAgent):
    """
    SARSA implementation with visit count-based learning rate adjustment.
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

    def observe(self, state, action, reward, next_state):
        """
        Update the Q-value using the SARSA update rule:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        """
        # Increment visit count
        self.visit_count[(state, action)] += 1

        # Dynamically adjust learning rate based on visit count
        visit_count = self.visit_count[(state, action)]
        adjusted_alpha = max(self.alpha_end, self.alpha / (1 + visit_count))  # Decay alpha as visits increase

        # Choose the next action using epsilon-greedy policy
        next_action = self.get_action(next_state)
        # Compute the SARSA update
        old_q = self.q_table[(state, action)]
        next_q = self.q_table[(next_state, next_action)]
        new_q = old_q + adjusted_alpha * (reward + self.gamma * next_q - old_q)
        self.q_table[(state, action)] = new_q

        # Update epsilon or any other time-dependent parameters
        self.update_learning_rate()