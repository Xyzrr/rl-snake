import gym
from gym import spaces
import numpy as np
import random

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if not IN_COLAB:
    from gym.envs.classic_control import rendering


class SnakeEnv(gym.Env):
    WIDTH = HEIGHT = 600
    IN_COLAB = IN_COLAB

    def __init__(self, board_size=8, food_reward=2, death_reward=-1, super_random_training=False, easy_world=False):
        self.board_size = board_size
        self.food_reward = food_reward
        self.death_reward = death_reward
        self.super_random_training = super_random_training
        self.easy_world = easy_world
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, shape=(9, 9, 3), dtype=np.uint8)

        self.viewer = None
        self.tile_width = self.WIDTH // self.board_size
        self.tile_height = self.HEIGHT // self.board_size

        self.reset()

    def player_out_of_bounds(self):
        x, y = self.player[-1]
        return not (0 <= x < self.board_size and 0 <= y < self.board_size)

    def collided_with_tail(self):
        return self.player[-1] in self.player[:-1]

    def player_on_food(self):
        x, y = self.player[-1]
        return (x == self.food[0]
                and y == self.food[1])

    def randomize_food(self):
        if self.easy_world:
            opts = [[3, 3], [3, 5], [5, 3], [5, 5]]
            self.food = random.choice(opts)
        else:
            while True:
                self.food = (random.randint(
                            0, self.board_size - 1), random.randint(0, self.board_size - 1))
                if self.food not in self.player:
                    break

    def add_square(self, viewer, position, color, margin=2):
        x, y = position
        l, r, t, b = x * self.tile_width + margin, (x + 1) * self.tile_width - margin, (
            self.board_size - y) * self.tile_height - margin, (self.board_size - y - 1) * self.tile_height + margin
        square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        square.set_color(*color)
        self.viewer.add_onetime(square)

    def render(self):
        if self.IN_COLAB:
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.WIDTH, self.HEIGHT)

        background = rendering.FilledPolygon(
            [(0, 0), (0, self.HEIGHT), (self.WIDTH, self.HEIGHT), (self.WIDTH, 0)])
        background.set_color(0, 0, 0)
        self.viewer.add_onetime(background)

        self.add_square(self.viewer, self.food, (.8, 0, 0))
        for i, node in enumerate(self.player):
            if i == 0 and len(self.player) > 1:
                self.add_square(self.viewer, node, (.3, .5, 0))
            elif i == len(self.player) - 1:
                self.add_square(self.viewer, node, (.7, 1, 0))
            else:
                self.add_square(self.viewer, node, (.5, .8, 0))

        return self.viewer.render()

    def get_observation(self):
        observation = np.zeros((self.board_size, self.board_size, 3))
        observation[self.food[1], self.food[0], 0] = 1
        for i, node in enumerate(self.player):
            if i == len(self.player) - 1:
                observation[node[1], node[0], 1] = 1
            else:
                observation[node[1], node[0], 2] = i + 1
        return observation

    def step(self, action):
        assert self.action_space.contains(action)

        current_position = self.player[-1]

        if action == 0:
            new_position = (current_position[0], current_position[1] - 1)
        elif action == 1:
            new_position = (current_position[0] + 1, current_position[1])
        elif action == 2:
            new_position = (current_position[0], current_position[1] + 1)
        elif action == 3:
            new_position = (current_position[0] - 1, current_position[1])

        self.player.append(new_position)

        reward = 0

        if self.player_on_food():
            self.size += 1
            reward = self.food_reward
            self.randomize_food()

        if len(self.player) > self.size:
            self.player.pop(0)

        done = False
        if self.player_out_of_bounds() or self.collided_with_tail():
            done = True
            reward = self.death_reward

        return np.zeros((self.board_size, self.board_size, 3)) if done else self.get_observation(), reward, done, {}

    def is_safe_spot(self, pos):
        x, y = pos
        return pos not in self.player and (0 <= x < self.board_size and 0 <= y < self.board_size)

    def generate_random_snake(self):
        self.size = random.randint(1, 20)
        self.player = [(random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1))]
        for i in range(self.size - 1):
            options = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            random.shuffle(options)
            for opt in options:
                new_pos = (self.player[-1][0] + opt[0], self.player[-1][1] + opt[1])
                if self.is_safe_spot(new_pos):
                    self.player.append(new_pos)
                    break
            else:
                break
        self.player.reverse()

    def reset(self):
        if self.super_random_training:
            self.generate_random_snake()
            self.randomize_food()
        else:
            if self.easy_world:
                self.player = [random.choice([[3, 3], [3, 5], [5, 3], [5, 5]])]
                # self.player = [(random.randint(
                # 0, self.board_size - 1), random.randint(0, self.board_size - 1))]
                self.randomize_food()
            else:
                self.player = [(random.randint(
                0, self.board_size - 1), random.randint(0, self.board_size - 1))]
                self.randomize_food()

            self.size = 1

        return self.get_observation()
