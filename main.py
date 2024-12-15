import argparse
import os
from training.train_single import train_agent
from training.test_single import test_agent
from training.train_double import train_double_agent
from training.test_double import test_double_agent
from environments.single_paddle_env import SinglePaddleEnv
from environments.pong_environment import MultiplayerPongEnv
from agents.qlearning_angent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from utils.parameters import REWARD_Values
from utils.plotter import plot_metrics


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Initialize environment
    env = MultiplayerPongEnv()
    env.render_mode = True

    left_agent = QLearningAgent()

    right_agent = QLearningAgent()

    # Train agents

    # After training double agents
    left_rewards, right_rewards = train_double_agent(
        env, left_agent, right_agent, episodes=10000, log_interval=1000, plot_path=None
    )

    # Prepare the metrics dictionary for plotting
    metrics_dict = {
        "Left Rewards": left_rewards,
        "Right Rewards": right_rewards
    }

    # Call the plot method
    plot_metrics(
        metrics_dict=metrics_dict,
        rolling_window=100,
        title="Training Rewards for Left and Right Agents",
        xlabel="Episodes",
        ylabel="Rewards",
        save_path="results/double_agent_training_rewards.png"
    )

    # Test agents
    left_rewards, right_rewards = test_double_agent(
        env, left_agent, right_agent, episodes=100, render=True, log_interval=10, plot_path=None
    )

    # Prepare the metrics dictionary for plotting
    metrics_dict = {
        "Left Rewards": left_rewards,
        "Right Rewards": right_rewards
    }

    # Call the plot method
    plot_metrics(
        metrics_dict=metrics_dict,
        rolling_window=10,
        title="Testing Rewards for Left and Right Agents",
        xlabel="Episodes",
        ylabel="Rewards",
        save_path="results/double_agent_testing_rewards.png"
    )

    # Close environment
    env.close()