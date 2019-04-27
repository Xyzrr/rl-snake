import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
import random


class SnakeEnv(gym.Env):
    WIDTH = HEIGHT = 600

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (spaces.Box(low=0, high=15, shape=(2,), dtype=np.int32), spaces.Box(low=-16, high=16, shape=(2,), dtype=np.int32)))

        self.viewer = None
        self.tile_width = self.WIDTH // self.board_size
        self.tile_height = self.HEIGHT // self.board_size

        self.reset()

    def player_out_of_bounds(self):
        return not (0 <= self.player[0] < self.board_size and 0 <= self.player[1] < self.board_size)

    def player_on_food(self):
        return (self.player[0] == self.food[0]
                and self.player[1] == self.food[1])

    def add_square(self, viewer, x, y, color):
        l, r, t, b = x * \
            self.tile_width, (x + 1) * self.tile_width, y * \
            self.tile_height, (y + 1) * self.tile_height
        square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        square.set_color(*color)
        self.viewer.add_onetime(square)

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.WIDTH, self.HEIGHT)

        self.add_square(self.viewer, self.player[0], self.player[1], (0, 0, 0))
        self.add_square(self.viewer, self.food[0], self.food[1], (200, 0, 0))

        return self.viewer.render()

    def get_observation(self):
        return ((self.player[0], self.player[1]), (self.food[0], self.food[1]))

    def step(self, action):
        assert self.action_space.contains(action)

        if action == 0:
            self.player[1] += 1
        elif action == 1:
            self.player[0] += 1
        elif action == 2:
            self.player[1] -= 1
        elif action == 3:
            self.player[0] -= 1

        reward = -0.1

        if self.player_on_food():
            self.size += 1
            reward = 10
            self.food = [random.randint(
                0, self.board_size - 1), random.randint(0, self.board_size - 1)]

        done = False
        if self.player_out_of_bounds():
            done = True
            reward = -1

        return self.get_observation(), reward, done, {}

    def reset(self):
        self.player = [random.randint(
            0, self.board_size - 1), random.randint(0, self.board_size - 1)]
        self.food = [random.randint(
            0, self.board_size - 1), random.randint(0, self.board_size - 1)]
        self.size = 1
        return self.get_observation()
