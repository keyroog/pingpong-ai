from agents.base_agent import BaseAgent
from utils.parameters import SARSA_Parameters, _ACTIONS

class SARSAAgent(BaseAgent):
    """
    SARSA implementation with visit count-based learning rate adjustment.
    """
    def __init__(self, **kwargs):
        """
        Initialize the SARSA agent with default or provided parameters.
        :param kwargs: Parameters to override defaults from SARSA_Parameters.
        """
        params = SARSA_Parameters.copy()  # Use default parameters
        params.update(kwargs)  # Override defaults with provided arguments
        super().__init__(actions=_ACTIONS, **params)

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
