import os
from agents.qlearning_angent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from environments.pong_environment import MultiplayerPongEnv
from training.train_double import train_double_agent
from utils.parameters import Q_Parameters, SARSA_Parameters
from utils.plotter import plot_metrics

def train_and_save_models(left_agent_type="qlearning", right_agent_type="sarsa", episodes=5000):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Initialize environment
    env = MultiplayerPongEnv()
    env.render_mode = False  # Disable rendering for faster training

    # Initialize agents
    left_agent = QLearningAgent(**Q_Parameters) if left_agent_type == "qlearning" else SARSAAgent(**SARSA_Parameters)
    right_agent = QLearningAgent(**Q_Parameters) if right_agent_type == "qlearning" else SARSAAgent(**SARSA_Parameters)

    # Train agents
    left_rewards, right_rewards = train_double_agent(
        env, left_agent, right_agent, episodes=episodes, log_interval=500
    )

    plot_metrics(
        metrics_dict={"Left Rewards": left_rewards, "Right Rewards": right_rewards},
        rolling_window=100,
        title="Training Rewards",
        xlabel="Episodes",
        ylabel="Rewards",
        save_path=f"results/{left_agent_type}_vs_{right_agent_type}_training_rewards_{episodes}.png",
    )

    # Save the trained models
    left_model_path = f"models/{left_agent_type}_models/{left_agent_type}_{episodes}k_left.pkl"
    right_model_path = f"models/{right_agent_type}_models/{right_agent_type}_{episodes}k_right.pkl"
    left_agent.save(left_model_path)
    right_agent.save(right_model_path)

    print(f"Models saved:\n  Left agent: {left_model_path}\n  Right agent: {right_model_path}")

    # Close environment
    env.close()

train_and_save_models(left_agent_type="sarsa", right_agent_type="sarsa", episodes=500000)
