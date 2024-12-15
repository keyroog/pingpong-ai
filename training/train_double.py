
def train_double_agent(env, agent_left, agent_right, episodes, log_interval=100, plot_path=None):
    """
    Train two agents simultaneously in the environment.

    :param env: The multiplayer environment.
    :param agent_left: Left paddle agent.
    :param agent_right: Right paddle agent.
    :param episodes: Number of episodes for training.
    :param log_interval: Interval for logging progress.
    :param plot_path: Path to save training rewards plot.
    :return: Tuple of (left_rewards, right_rewards).
    """
    left_rewards = []
    right_rewards = []

    for episode in range(episodes):
        state = env.reset()
        left_total_reward = 0
        right_total_reward = 0
        done = False

        while not done:
            # Choose actions for both agents
            left_action = agent_left.get_action(state)
            right_action = agent_right.get_action(state)

            # Take actions in the environment
            next_state, (left_reward, right_reward), done, _ = env.step((left_action, right_action))

            # Update both agents
            agent_left.observe(state, left_action, left_reward, next_state)
            agent_right.observe(state, right_action, right_reward, next_state)

            # Accumulate rewards
            left_total_reward += left_reward
            right_total_reward += right_reward

            # Move to the next state
            state = next_state

        left_rewards.append(left_total_reward)
        right_rewards.append(right_total_reward)

        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_left = sum(left_rewards[-log_interval:]) / log_interval
            avg_right = sum(right_rewards[-log_interval:]) / log_interval
            print(f"Episode {episode + 1}: Avg Left Reward: {avg_left}, Avg Right Reward: {avg_right}")

    return left_rewards, right_rewards