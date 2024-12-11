import pygame
import matplotlib.pyplot as plt
import numpy as np
import sys

class Visualizer:
    """
    Classe per visualizzare l'ambiente Pong.
    """
    def __init__(self, width=500, height=500, ball_radius=8, paddle_width=8, paddle_height_ratio=0.2):
        """
        :param width: Larghezza della finestra.
        :param height: Altezza della finestra.
        :param ball_radius: Raggio della palla in pixel.
        :param paddle_width: Larghezza delle racchette in pixel.
        :param paddle_height_ratio: Altezza delle racchette rispetto all'altezza della finestra.
        """
        self.width = width
        self.height = height
        self.ball_radius = ball_radius
        self.paddle_width = paddle_width
        self.paddle_height = int(height * paddle_height_ratio)

        self.window = None
        self.clock = None

    def render(self, ball_pos, left_paddle_y, right_paddle_y,multiplayer=False):
        """
        Renderizza l'ambiente.
        :param ball_pos: Posizione della palla (x, y) in coordinate normalizzate.
        :param left_paddle_y: Posizione della racchetta sinistra (margine superiore) in coordinate normalizzate.
        :param right_paddle_y: Posizione della racchetta destra (margine superiore) in coordinate normalizzate.
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong Environment")
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Disegna la palla
        ball_pos = int(ball_pos[0] * self.width), int(ball_pos[1] * self.height)
        pygame.draw.circle(self.window, (255, 255, 255), ball_pos, self.ball_radius)

        # Disegna la racchetta sinistra
        left_paddle_rect = pygame.Rect(
            0,
            int(left_paddle_y * self.height),
            self.paddle_width,
            self.paddle_height,
        )
        pygame.draw.rect(self.window, (255, 255, 255), left_paddle_rect)

        # Disegna la racchetta destra
        if multiplayer:
          right_paddle_rect = pygame.Rect(
              self.width - self.paddle_width,
              int(right_paddle_y * self.height),
              self.paddle_width,
              self.paddle_height,
          )
          pygame.draw.rect(self.window, (255, 255, 255), right_paddle_rect)

        # Aggiorna il display
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        """
        Chiude la finestra di rendering.
        """
        if self.window:
            pygame.quit()
            self.window = None
            sys.exit()