from environments.base_environment import BasePongEnv
from utils.parameters import REWARD_Values
import numpy as np
class SinglePaddleEnv(BasePongEnv):
    def __init__(self):
        super(SinglePaddleEnv, self).__init__()
        self.score = 0

    def step(self, action):
        if self.done:
            raise RuntimeError("You need to reset the environment before calling step().")

        self.left_paddle_y += self.action_list[action]
        self._clamp_paddle_positions()

        self._update_ball_position()

        reward = 0
        if self.ball_x <= 0:  # Collision with left paddle
            if self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height:
                self._handle_paddle_collision(self.left_paddle_y)
                reward = 1
                self.score += 1
            else:
                self.done = True
                reward = -1

        if self.ball_x >= 1:  # Collision with right wall
            self.velocity_x *= -1

        return self._get_discretized_state(), reward, self.done, {"score": self.score}

    def reset(self):
        self.score = 0
        return super(SinglePaddleEnv, self).reset()