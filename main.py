from agents.qlearning_angent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from environments.pong_environment import MultiplayerPongEnv
from training.train_double import train_double_agent
from training.test_double import test_double_agent
from utils.plotter import plot_metrics
import pygame

def start_main(config):
    # Initialize environment
    pygame.init()  # Inizializza Pygame una sola volta
    env = MultiplayerPongEnv()
    user_mode = config["mode"] == "agent_vs_player"  # True if user is playing against agent
    print(f"config: {config}")
    print(f"User Mode: {user_mode}")

    # Initialize left agent
    if config["train_new"]:
        left_agent = QLearningAgent(**config["left_agent_params"]) if config["left_agent_type"] == "qlearning" else SARSAAgent(**config["left_agent_params"])
    else:
        left_agent = QLearningAgent() if config["left_agent_type"] == "qlearning" else SARSAAgent()
        left_agent.load(f"models/{config['left_agent_type']}_models/{config['left_model']}")  # Load pre-trained left model

    # Initialize right agent if in agent-vs-agent mode
    if config["mode"] == "agent_vs_agent":
        if config["train_new"]:
            right_agent = QLearningAgent(**config["right_agent_params"]) if config["right_agent_type"] == "qlearning" else SARSAAgent(**config["right_agent_params"])
        else:
            right_agent = QLearningAgent() if config["right_agent_type"] == "qlearning" else SARSAAgent()
            right_agent.load(f"models/{config['right_agent_type']}_models/{config['right_model']}")  # Load pre-trained right model
    else:
        right_agent = None  # Player-controlled opponent

    if config["train_new"]:
        # Train agents
        #if agent_vs_player is selected the right agent is the clone of the left agent for the training
        if config["mode"] == "agent_vs_player":
            right_agent = QLearningAgent(**config["left_agent_params"]) if config["left_agent_type"] == "qlearning" else SARSAAgent(**config["left_agent_params"])

        left_rewards, right_rewards = train_double_agent(
            env,
            left_agent,
            right_agent,
            episodes=config["episodes"],
            log_interval=1000,
            plot_path="results/training_rewards.png",
            user_mode=user_mode,
        )

        # Save Q-tables after training
        left_model_path = f"models/{config['left_agent_type']}_models/{config['left_agent_type']}_{config["episodes"]}_left.pkl"
        left_agent.save(left_model_path)

        if right_agent:
            right_model_path = f"models/{config['right_agent_type']}_models/{config['right_agent_type']}_{config["episodes"]}_right.pkl"
            right_agent.save(right_model_path)

        print(f"Models saved:\n  Left Agent: {left_model_path}\n  Right Agent: {right_model_path if right_agent else 'None'}")


        save_path = f"results/{config['left_agent_type']}_vs_{config['right_agent_type']}_training_rewards_{config['episodes']}.png"
        # Plot training results
        plot_metrics(
            metrics_dict={"Left Rewards": left_rewards, "Right Rewards": right_rewards},
            rolling_window=100,
            title="Training Rewards",
            xlabel="Episodes",
            ylabel="Rewards",
            save_path=save_path,
        )

        test_double_agent(
            env,
            left_agent,
            right_agent,
            episodes=100,
            render=True,
            log_interval=10,
            plot_path="results/testing_rewards.png",
            user_mode=config["mode"] == "user_vs_agent",
        )
    else:
        # Test agents with pre-trained models
        left_rewards, right_rewards = test_double_agent(
            env,
            left_agent,
            right_agent,
            episodes=100,
            render=True,
            log_interval=10,
            plot_path="results/testing_rewards.png",
            user_mode=user_mode,
        )

        # Plot testing results
        plot_metrics(
            metrics_dict={"Left Rewards": left_rewards, "Right Rewards": right_rewards},
            rolling_window=10,
            title="Testing Rewards",
            xlabel="Episodes",
            ylabel="Rewards",
            save_path="results/double_agent_testing_rewards.png",
        )

    # Close environment
    env.close()
    # Close Pygame
    pygame.quit()



if __name__ == "__main__":
    from gui.game_config_gui import launch_game_config_gui  # Import your updated GUI code
    launch_game_config_gui()