def test_double_agent(env, agent_left, agent_right, episodes, render=False, log_interval=10, plot_path=None, user_mode=False):
    """
    Test two agents in a multiplayer environment.

    :param env: The multiplayer environment.
    :param agent_left: Left paddle agent.
    :param agent_right: Right paddle agent.
    :param episodes: Number of episodes for testing.
    :param render: Whether to render the environment during testing.
    :param log_interval: Interval for logging progress.
    :param plot_path: Path to save testing rewards plot.
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
            # Agents choose their actions (exploitation only)
            left_action = agent_left.get_best_action(state)
            if(not user_mode):
                right_action = agent_right.get_best_action(state)
            else:
                right_action = env._get_user_action()

            # Step the environment with both actions
            next_state, (left_reward, right_reward), done, _ = env.step((left_action, right_action))

            # Accumulate rewards
            left_total_reward += left_reward
            right_total_reward += right_reward

            # Move to the next state
            state = next_state

            # Render the environment if enabled
            if render:
                env.render()

        left_rewards.append(left_total_reward)
        right_rewards.append(right_total_reward)

        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_left = sum(left_rewards[-log_interval:]) / log_interval
            avg_right = sum(right_rewards[-log_interval:]) / log_interval
            print(f"Episode {episode + 1}: Avg Left Reward: {avg_left}, Avg Right Reward: {avg_right}")

    return left_rewards, right_rewards