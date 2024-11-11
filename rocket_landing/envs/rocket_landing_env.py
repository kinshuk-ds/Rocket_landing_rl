import gym
from gym import spaces
import pygame
import numpy as np
import time
import random

class RocketLandingEnv(gym.Env):
    def __init__(self):
        super(RocketLandingEnv, self).__init__()

        # Rocket parameters
        self.screen_width = 800
        self.screen_height = 600
        self.rocket_width = 20
        self.rocket_height = 40
        self.gravity = 0.2
        self.damping = 0.99
        self.thrust = 0.5
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.ORANGE = (255, 165, 0)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)

        # Define the action and observation space
        self.action_space = spaces.Discrete(3)  # Actions: 0=thrust, 1=rotate left, 2=rotate right
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Rocket Landing with DQN")
        self.clock = pygame.time.Clock()

        self.rocket_surface = pygame.Surface((self.rocket_width, self.rocket_height), pygame.SRCALPHA)
        pygame.draw.rect(self.rocket_surface, self.BLACK, (0, 0, self.rocket_width, self.rocket_height))

        self.reset()

    def reset(self):
        # Reset rocket state
        self.rocket_pos = np.array([self.screen_width // 2, 50], dtype=float)
        self.rocket_velocity = np.array([0, 1], dtype=float)
        self.rocket_angle = 0
        self.landed = False
        self.crashed = False

        return self._get_observation()

    def _get_observation(self):
        # Return current state of the rocket
        return np.array([
            self.rocket_pos[0],         # Rocket X position
            self.rocket_pos[1],         # Rocket Y position
            self.rocket_velocity[0],    # Rocket X velocity
            self.rocket_velocity[1],    # Rocket Y velocity
            self.rocket_angle           # Rocket angle
        ], dtype=np.float32)

    def step(self, action):
        thrusting = False

        # Action: 0 = thrust, 1 = rotate left, 2 = rotate right
        if action == 0:  # Apply thrust
            self.rocket_velocity[1] -= self.thrust * np.cos(np.radians(self.rocket_angle))
            self.rocket_velocity[0] += self.thrust * np.sin(np.radians(self.rocket_angle))
            thrusting = True
        elif action == 1:  # Rotate counterclockwise
            self.rocket_angle += 2
        elif action == 2:  # Rotate clockwise
            self.rocket_angle -= 2

        # Apply gravity and damping
        self.rocket_velocity[1] += self.gravity
        self.rocket_velocity[0] *= self.damping
        self.rocket_pos += self.rocket_velocity

        # Check if the rocket has landed or crashed
        reward = -1  # Small penalty for every step
        done = False
        if self.rocket_pos[1] >= self.screen_height - self.rocket_height:
            self.rocket_pos[1] = self.screen_height - self.rocket_height
            if abs(self.rocket_velocity[1]) <= 2.0 and abs(self.rocket_angle) <= 10:
                self.landed = True
                reward = 100  # Reward for safe landing
            else:
                self.crashed = True
                reward = -100  # Penalty for crash
            done = True

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        # Visual rendering of the rocket's current state
        self.screen.fill(self.WHITE)
        if self.crashed:
            self.draw_crash(self.rocket_pos)
        elif self.landed:
            self.draw_rotated_rocket(self.rocket_surface, self.rocket_pos, self.rocket_angle)
            self.draw_success(self.rocket_pos)
        else:
            self.draw_rotated_rocket(self.rocket_surface, self.rocket_pos, self.rocket_angle)
            if abs(self.rocket_velocity[1]) > 0:  # Show ignition if there’s thrust
                self.draw_ignition(self.rocket_pos)
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 frames per second

    def draw_rotated_rocket(self, surface, pos, angle):
        # Rotate the rocket and render it
        rotated_surface = pygame.transform.rotate(surface, -angle)
        rotated_rect = rotated_surface.get_rect(center=(pos[0] + self.rocket_width // 2, pos[1] + self.rocket_height // 2))
        self.screen.blit(rotated_surface, rotated_rect.topleft)

    def draw_ignition(self, pos):
        # Draw rocket ignition/flames at the bottom
        flame_pos = (pos[0] + self.rocket_width // 2 - 4, pos[1] + self.rocket_height)
        flame_width = random.randint(8, 12)
        flame_height = random.randint(12, 20)
        pygame.draw.ellipse(self.screen, self.ORANGE, (*flame_pos, flame_width, flame_height))
        pygame.draw.ellipse(self.screen, self.RED, (flame_pos[0] + 2, flame_pos[1] + 8, 4, 8))

    def draw_crash(self, pos):
        # Draw explosion effects when the rocket crashes
        for i in range(5, 50, 5):
            pygame.draw.circle(self.screen, self.RED, (int(pos[0] + self.rocket_width // 2), int(pos[1] + self.rocket_height // 2)), i, 1)
            pygame.draw.circle(self.screen, self.ORANGE, (int(pos[0] + self.rocket_width // 2), int(pos[1] + self.rocket_height // 2)), i // 2, 1)
            pygame.draw.circle(self.screen, self.YELLOW, (int(pos[0] + self.rocket_width // 2), int(pos[1] + self.rocket_height // 2)), i // 4, 1)
        font = pygame.font.Font(None, 36)
        text = font.render("Rocket Crashed!", True, self.RED)
        self.screen.blit(text, (pos[0] - 50, pos[1] - 50))

    def draw_success(self, pos):
        # Draw success message when the rocket lands safely
        font = pygame.font.Font(None, 36)
        text = font.render("Successful Landing!", True, self.GREEN)
        self.screen.blit(text, (pos[0] - 50, pos[1] - 50))

    def close(self):
        # Close pygame window
        pygame.quit()
