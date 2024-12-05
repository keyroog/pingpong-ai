from utils.plotter import plot_rewards

def test_agent(env, agent, episodes, plot_path=None, render=False):
    """
    Test the trained agent in the environment.

    :param env: The environment instance.
    :param agent: The agent instance.
    :param episodes: Number of testing episodes.
    :param plot_path: Path to save testing rewards plot.
    :return: List of total rewards per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Exploit only: choose the best action
            action = agent.get_best_action(state)

            # Take action in the environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if render:
              env.render()

            # Update state
            state = next_state

        total_rewards.append(total_reward)

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Test Reward: {avg_reward}")

    # Plot rewards if plot_path is provided
    if plot_path:
        plot_rewards(rewards=total_rewards, rolling_window=10, save_path=plot_path)

    return total_rewards