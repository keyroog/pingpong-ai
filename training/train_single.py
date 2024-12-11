from utils.plotter import plot_rewards

def train_agent(env, agent, episodes, log_interval=500, plot_path=None):
    """
    Train an agent in the given environment.

    :param env: The environment instance.
    :param agent: The agent instance.
    :param episodes: Number of training episodes.
    :param log_interval: Interval for logging progress.
    :param plot_path: Path to save training rewards plot.
    :return: List of total rewards per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose an action using the agent's policy
            action = agent.get_action(state)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update the agent with the observed experience
            agent.observe(state, action, reward, next_state)

            # Move to the next state
            state = next_state

        total_rewards.append(total_reward)

        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = sum(total_rewards[-log_interval:]) / log_interval
            print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")

    # Plot rewards if plot_path is provided
    if plot_path:
        plot_rewards(rewards=total_rewards, rolling_window=100, save_path=plot_path)

    return total_rewards