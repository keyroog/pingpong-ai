from utils.visualizer import Visualizer
from utils.discretizer import Discretizer
import gym
from gym import spaces
import numpy as np
import random

class BasePongEnv(gym.Env):
    """
    Classe base per gli ambienti Pong con visualizzazione e discretizzazione integrate.
    """
    def __init__(self, paddle_height=0.2, render_mode=False):
        super(BasePongEnv, self).__init__()
        # Parametri del campo
        self.field_width = 1.0
        self.field_height = 1.0
        self.paddle_height = paddle_height
        self.ball_radius = 0.02

        self.min_speed = 0.03
        self.max_speed = 0.04
        self.angle_limit = 0.7 

        # Azioni
        self.action_list = [0, 0.04, -0.04]

        # Spazi di osservazione e azione
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0]),
            high=np.array([1, 1, 1, 1, 1 - self.paddle_height, 1 - self.paddle_height]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Discretizzazione
        bins_per_dimension = [12, 12, 2, 2, 12, 12]
        self.discretizer = Discretizer(bins_per_dimension=bins_per_dimension)

        # Visualizzazione
        self.render_mode = render_mode
        self.visualizer = Visualizer()

        # Stato iniziale
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.ball_x = self.field_width / 2
        self.ball_y = self.field_height / 2

        # Randomize ball's initial velocity within reasonable limits
        self.velocity_x = random.choice([0.02, -0.02])
        self.velocity_y = random.uniform(-0.02, 0.02)

        self.left_paddle_y = (self.field_height - self.paddle_height) / 2
        self.right_paddle_y = (self.field_height - self.paddle_height) / 2
        self.done = False

        return self._get_discretized_state()
        
    def step(self, actions):
        """
        Aggiorna lo stato in base all'azione. Da implementare nelle classi derivate.
        """
        raise NotImplementedError("Il metodo 'step' deve essere implementato nelle classi derivate.")

    def _update_ball_position(self):
        """
        Updates the ball's position and handles wall collisions.
        """
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y

        # Collision with top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.field_height:
            self.velocity_y = self._reflect_velocity(self.velocity_y, axis='vertical')

        # Enforce minimum speed
        self._enforce_min_speed()

    def _handle_paddle_collision(self, paddle_y):
        """
        Handles the collision between the ball and a paddle.
        """
        self.velocity_x = self._reflect_velocity(self.velocity_x, axis='horizontal')

        # Compute impact factor and adjust velocity
        paddle_center = paddle_y + self.paddle_height / 2
        impact_factor = (self.ball_y - paddle_center) / (self.paddle_height / 2)
        impact_factor = np.clip(impact_factor, -1, 1)

        self.velocity_y += impact_factor * 0.02
        self.velocity_x += random.uniform(-0.002, 0.002)

        # Enforce angle and speed limits
        self._clamp_angle()
        self._enforce_min_speed()

    def _reflect_velocity(self, velocity, random_adjustment=0.003, axis='horizontal'):
        """
        Reflects velocity and ensures it adheres to the speed limits.
        
        :param velocity: The velocity to reflect (either horizontal or vertical).
        :param random_adjustment: Small randomness added to the velocity after reflection.
        :param axis: 'horizontal' for x-axis reflection, 'vertical' for y-axis reflection.
        :return: Adjusted velocity.
        """
        velocity *= -1  # Reverse the direction of the velocity
        velocity += random.uniform(-random_adjustment, random_adjustment)

        # Ensure the ball's position stays within bounds
        if axis == 'horizontal':
            self.ball_x = max(0, min(self.field_width, self.ball_x))
        elif axis == 'vertical':
            self.ball_y = max(0, min(self.field_height, self.ball_y))

        # Clamp the velocity within the allowed limits
        return np.clip(velocity, -self.max_speed, self.max_speed)

    def _enforce_min_speed(self):
        """
        Ensures that the ball maintains a minimum speed.
        """
        speed = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if speed < self.min_speed:
            scale = self.min_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale

    def _clamp_angle(self):
        """
        Prevents the ball from becoming too oblique by clamping the angle.
        """
        ratio = abs(self.velocity_y / self.velocity_x)
        if ratio > self.angle_limit:
            scale = self.angle_limit / ratio
            self.velocity_y *= scale

    def _clamp_paddle_positions(self):
        """
        Mantiene le racchette entro i limiti del campo.
        """
        self.left_paddle_y = np.clip(self.left_paddle_y, 0, self.field_height - self.paddle_height)
        self.right_paddle_y = np.clip(self.right_paddle_y, 0, self.field_height - self.paddle_height)

    def _get_continuous_state(self):
        """
        Restituisce lo stato continuo.
        """
        return np.array([
            self.ball_x,
            self.ball_y,
            self.velocity_x,
            self.velocity_y,
            self.left_paddle_y,
            self.right_paddle_y,
        ], dtype=np.float32)

    def _get_discretized_state(self):
        """
        Restituisce lo stato discretizzato usando il Discretizer.
        """
        continuous_state = self._get_continuous_state()
        return self.discretizer.discretize(continuous_state)

    def render(self):
        """
        Renderizza l'ambiente usando il Visualizer.
        """
        if self.render_mode and self.visualizer:
            self.visualizer.render(
                ball_pos=(self.ball_x, self.ball_y),
                left_paddle_y=self.left_paddle_y,
                right_paddle_y=self.right_paddle_y
            )

    def close(self):
        """
        Chiude il visualizzatore, se esistente.
        """
        if self.render_mode and self.visualizer:
            self.visualizer.close()