from environments.base_environment import BasePongEnv
from utils.parameters import REWARD_Values
import numpy as np

class SinglePaddleEnv(BasePongEnv):
    def __init__(self):
        super(SinglePaddleEnv, self).__init__()
        self.score = 0
        self.speed_increment = 0.001  # Increment speed after each hit

    def step(self, action):
        """
        Perform a step in the environment based on the player's action.
        """
        if self.done:
            raise RuntimeError("You need to reset the environment before calling step().")

        # Update paddle position
        self.left_paddle_y += self.action_list[action]
        self._clamp_paddle_positions()

        # Update ball position
        self._update_ball_position()

        # Handle collisions
        reward = 0
        if self.ball_x <= 0:  # Collision with left paddle
            if self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height:
                self._handle_paddle_collision(self.left_paddle_y)
                reward = REWARD_Values["hit"]
                self.score += 1

                # Increase speed incrementally
                self.max_speed += self.speed_increment
            else:
                self.done = True
                reward = REWARD_Values["miss"]

        if self.ball_x >= 1:  # Collision with right wall
            self.velocity_x *= -1

        return self._get_discretized_state(), reward, self.done, {"score": self.score}

    def reset(self):
        """
        Reset the environment and player's score.
        """
        self.score = 0
        return super(SinglePaddleEnv, self).reset()