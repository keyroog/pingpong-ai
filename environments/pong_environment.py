from utils.visualizer import Visualizer
from utils.discretizer import Discretizer
import gym
from gym import spaces
import numpy as np
import random
from utils.parameters import REWARD_Values
import pygame

class MultiplayerPongEnv(gym.Env):
    """
    Multiplayer Pong environment for training two agents simultaneously.
    """
    def __init__(self):
        super(MultiplayerPongEnv, self).__init__()
        self.field_width = 1.0
        self.field_height = 1.0
        self.paddle_height = 0.2
        self.ball_radius = 0.02
        self.angle_limit = 0.7
        self.min_speed = 0.04

        # Discretization
        bins_per_dimension = [12, 12, 2, 2, 12, 12]
        self.discretizer = Discretizer(bins_per_dimension=bins_per_dimension)

        # Actions
        self.action_list = [0, 0.04, -0.04]

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0]),
            high=np.array([1, 1, 1, 1, 1 - self.paddle_height, 1 - self.paddle_height]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.visualizer = Visualizer()

        # Initial state
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.ball_x = self.field_width / 2
        self.ball_y = self.field_height / 2

        # Randomize ball's initial velocity
        self.velocity_x = random.choice([0.03, -0.03])
        self.velocity_y = random.uniform(-0.02, 0.02)

        self.left_paddle_y = (self.field_height - self.paddle_height) / 2
        self.right_paddle_y = (self.field_height - self.paddle_height) / 2
        self.done = False

        return self._get_discretized_state()

    def step(self, actions):
        """
        Updates the state based on both paddles' actions.
        :param actions: Tuple of (left_action, right_action).
        """
        left_action, right_action = actions

        # Update paddles
        self.left_paddle_y += self.action_list[left_action]
        self.right_paddle_y += self.action_list[right_action]

        # Clamp paddle positions
        self._clamp_paddle_positions()

        # Update ball position
        self._update_ball_position()

        # Handle collisions and rewards
        left_reward, right_reward = 0, 0
        if self.ball_x <= 0:  # Collision with left paddle
            if self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height:
                self._handle_paddle_collision(self.left_paddle_y)
                left_reward = 1
            else:
                self.done = True
                left_reward = -1
                right_reward = 1

        if self.ball_x >= 1:  # Collision with right paddle
            if self.right_paddle_y <= self.ball_y <= self.right_paddle_y + self.paddle_height:
                self._handle_paddle_collision(self.right_paddle_y)
                right_reward = 1
            else:
                self.done = True
                right_reward = -1
                left_reward = 1

        return self._get_discretized_state(), (left_reward, right_reward), self.done, {}

    def render(self):
        """
        Renders the environment.
        """
        if self.visualizer:
            self.visualizer.render(
                ball_pos=(self.ball_x, self.ball_y),
                left_paddle_y=self.left_paddle_y,
                right_paddle_y=self.right_paddle_y
            )

    def close(self):
        """
        Closes the visualizer, if any.
        """
        if self.visualizer:
            self.visualizer.close()

    def _get_ai_action(self):
        """
        Determines the AI's action based on the selected difficulty level.
        """
        if self.difficulty == "easy":
            return self._easy_ai()
        elif self.difficulty == "medium":
            return self._medium_ai()
        elif self.difficulty == "hard":
            return self._hard_ai()
        else:
            raise ValueError(f"Unknown difficulty level: {self.difficulty}")

    def _easy_ai(self):
        """
        Moves the paddle randomly or slightly towards the ball.
        """
        if random.random() < 0.5:
            return random.choice([1, 2])  # Randomly move up or down
        elif self.ball_y > self.right_paddle_y + self.paddle_height / 2:
            return 1  # Move down
        elif self.ball_y < self.right_paddle_y + self.paddle_height / 2:
            return 2  # Move up
        return 0  # Stay

    def _medium_ai(self):
        """
        Follows the ball with a slight delay.
        """
        target_position = self.ball_y
        if target_position > self.right_paddle_y + self.paddle_height / 2:
            return 1  # Move down
        elif target_position < self.right_paddle_y + self.paddle_height / 2:
            return 2  # Move up
        return 0  # Stay

    def _hard_ai(self):
        """
        Tracks the ball perfectly to act as an unbeatable opponent.
        """
        if self.ball_y > self.right_paddle_y + self.paddle_height / 2:
            return 1  # Move down
        elif self.ball_y < self.right_paddle_y + self.paddle_height / 2:
            return 2  # Move up
        return 0  # Stay

    def _update_ball_position(self):
        """
        Updates the ball's position and handles wall collisions.
        """
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y

        # Collision with top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.field_height:
            self.velocity_y *= -1  # Reflect vertical velocity

    def _handle_paddle_collision(self, paddle_y):
        """
        Handles the collision between the ball and a paddle.
        """
        self.velocity_x = -self.velocity_x

        paddle_center = paddle_y + self.paddle_height / 2
        impact_factor = (self.ball_y - paddle_center) / (self.paddle_height / 2)
        impact_factor = np.clip(impact_factor, -1, 1)

        self.velocity_y += impact_factor * 0.005

        speed_increase = 1.02
        self.velocity_x *= speed_increase

    def _clamp_paddle_positions(self):
        """
        Keeps the paddles within the bounds of the field.
        """
        self.left_paddle_y = np.clip(self.left_paddle_y, 0, self.field_height - self.paddle_height)
        self.right_paddle_y = np.clip(self.right_paddle_y, 0, self.field_height - self.paddle_height)

    def _get_continuous_state(self):
        """
        Returns the continuous state representation.
        """
        return np.array([
            self.ball_x,
            self.ball_y,
            self.velocity_x,
            self.velocity_y,
            self.left_paddle_y,
            self.right_paddle_y,
        ], dtype=np.float32)

    def _get_user_action(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            return 2  # Muovi verso l'alto
        elif keys[pygame.K_DOWN]:
            return 1  # Muovi verso il basso
        else:
            return 0  # Nessun movimento

    def _get_discretized_state(self):
        """
        Returns the discretized state representation using the Discretizer.
        """
        continuous_state = self._get_continuous_state()
        return self.discretizer.discretize(continuous_state)