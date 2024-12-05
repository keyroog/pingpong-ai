import argparse
import os
from training.train_single import train_agent
from training.test_single import test_agent
from environments.single_paddle_env import SinglePaddleEnv
from agents.qlearning_angent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from utils.parameters import REWARD_Values
from utils.plotter import plot_rewards


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Initialize environment
    env = SinglePaddleEnv()
    env.render_mode = True

    # Select agent
    agent_type = "qlearning"  # Change as needed
    if agent_type == "qlearning":
        agent = QLearningAgent()
    elif agent_type == "sarsa":
        agent = SARSAAgent()

    # Training
    print(f"Training {agent_type} agent for 1000 episodes...")
    train_rewards = train_agent(
        env,
        agent,
        episodes=100,
        plot_path="results/training_rewards_4.png"
    )

    # Testing
    print(f"Testing {agent_type} agent for 100 episodes...")
    test_rewards = test_agent(
        env,
        agent,
        episodes=100,
        plot_path="results/testing_rewards.png",
        render=True
    )

    plot_rewards(rewards=train_rewards, test_rewards=test_rewards, rolling_window=50, save_path="results/combined_rewards.png")

    # Close environment
    env.close()