import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards, test_rewards=None, rolling_window=50, save_path=None):
    """
    Plots training and testing rewards with optional rolling averages.
    """
    plt.figure(figsize=(12, 6))

    # Plot training rewards
    plt.plot(rewards, label="Training Rewards", alpha=0.6)
    if rolling_window > 1:
        rolling_avg = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode="valid")
        plt.plot(range(rolling_window - 1, len(rewards)), rolling_avg, 
                 label=f"Rolling Avg (Train, window={rolling_window})", color="orange")

    # Plot testing rewards if available
    if test_rewards:
        plt.plot(test_rewards, label="Testing Rewards", alpha=0.6, linestyle="--", color="green")
        if rolling_window > 1:
            rolling_avg_test = np.convolve(test_rewards, np.ones(rolling_window) / rolling_window, mode="valid")
            plt.plot(range(rolling_window - 1, len(test_rewards)), rolling_avg_test, 
                     label=f"Rolling Avg (Test, window={rolling_window})", color="red")

    # Add labels, title, and legend
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards Over Episodes")
    plt.legend()
    plt.grid()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_learning_rate(agent, state_actions, num_episodes, save_path=None):
    """
    Plot the learning rate progression for specific state-action pairs.
    """
    learning_rate_progression = {sa: [] for sa in state_actions}

    for episode in range(num_episodes):
        for sa in state_actions:
            state, action = sa
            visit_count = agent.visit_count.get(sa, 0)
            adjusted_alpha = max(agent.alpha_end, agent.alpha / (1 + visit_count))
            learning_rate_progression[sa].append(adjusted_alpha)

        # Simulate an update (increment visit counts for testing purposes)
        for sa in state_actions:
            agent.visit_count[sa] += 1

    # Plot results
    plt.figure(figsize=(10, 6))
    for sa, rates in learning_rate_progression.items():
        plt.plot(rates, label=f"State-Action: {sa}")
    plt.xlabel("Episodes")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Progression")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_state_visit_distribution(visit_count, save_path=None):
    """
    Plots the state-action visit distribution.
    """
    states = [str(state_action) for state_action in visit_count.keys()]
    visits = list(visit_count.values())

    plt.figure(figsize=(12, 6))
    plt.bar(states, visits, color="blue", alpha=0.7)
    plt.xlabel("State-Action Pairs")
    plt.ylabel("Visit Frequency")
    plt.title("State-Action Visit Distribution")
    plt.xticks(rotation=90)
    plt.grid(axis="y")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_q_value_convergence(q_values, save_path=None):
    """
    Plots the Q-value convergence for specific states.
    """
    plt.figure(figsize=(12, 6))
    for state, q_values_list in q_values.items():
        plt.plot(q_values_list, label=f"State: {state}")

    plt.xlabel("Episodes")
    plt.ylabel("Q-Value")
    plt.title("Q-Value Convergence")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()