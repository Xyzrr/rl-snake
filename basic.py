import gym
import gym_snake
import time
import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


wandb.init(project="snake")
wandb.config.discount_rate = .6
wandb.config.initial_eps = 1
wandb.config.decay_factor = .9998
wandb.config.board_size = 5
wandb.config.food_reward = 1
wandb.config.death_reward = -1

debug = False
no_render = False
auto_play = False
fast_forward_remaining = 0


def render_env_until_key_press(env):
    global fast_forward_remaining, auto_play
    waiting = True

    def key_press(key, mod):
        nonlocal waiting
        global no_render, fast_forward_remaining, debug, auto_play

        if (key == 112):  # p
            auto_play = not auto_play
            print(bcolors.OKBLUE +
                  'Turned {} autoplay'.format('on' if auto_play else 'off') + bcolors.ENDC)

        if (key == 100):  # d
            debug = not debug
            print(bcolors.OKBLUE +
                  'Turned {} debug'.format('on' if debug else 'off') + bcolors.ENDC)
            return

        if (key == 65519 or key == 65520 or key == 65507):  # option or ctrl
            return

        if (key == 99 and mod == 2):  # ctrl + c
            exit(0)

        if (49 <= key <= 57):  # 1-9
            digit = key - 48
            fast_forward_remaining = 10**digit - 1
            if mod == 132:  # option
                no_render = True
            else:
                no_render = False
            print(bcolors.OKBLUE + "Fast forwarding {} steps {}".format(
                fast_forward_remaining + 1, '(no render)' if no_render else '') + bcolors.ENDC)

        waiting = False

    if fast_forward_remaining > 0:
        fast_forward_remaining -= 1
        if not no_render:
            env.render()
    elif auto_play:
        time.sleep(0.2)
        env.render()
    else:
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        while waiting:
            time.sleep(0.01)
            env.render()
            pass


def print_step_before_move(step, observation, prediction, action, was_random):
    if not debug:
        return

    print()
    print('-' * 40, step, '-' * 40)
    print('Observation:             ')
    print(observation)
    print('Prediction:              ', prediction)

    action_str = '{} ({})'.format(
        action, ['up', 'right', 'down', 'left'][action])
    if was_random:
        print('Random move:             ', action_str)
    else:
        print('Move:                    ', action_str)


def print_step_after_move(reward, target_action_score, label, new_prediction):
    if not debug:
        return

    print()
    print('Reward:                  ', reward)
    print('Target action score:     ', target_action_score)
    print('Label:                   ', label)
    print('New prediction:          ', new_prediction)
    print('-' * 83)
    print()


def train(env, model, num_episodes=500):
    discount_rate = wandb.config.discount_rate
    eps = wandb.config.initial_eps
    decay_factor = wandb.config.decay_factor
    # counter = []
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        observation = env.reset()
        eps *= decay_factor
        done = False
        total_food = 0
        step = 0
        # wandb_table_data = []
        while not done:
            model_input = np.array([observation.reshape(5, 5, 1)])
            prediction = model.predict(model_input)
            if np.random.random() < eps:
                action = np.random.randint(0, 4)
                was_random = True
            else:
                action = np.argmax(prediction)
                was_random = False
            # counter.append(4**4 * action + 4**3 * player[0] + 4**2 * player[1] +
            #                4 * food[0] + food[1])
            print_step_before_move(
                step, observation, prediction, action, was_random)

            render_env_until_key_press(env)

            new_observation, reward, done, _ = env.step(action)
            # wandb_table_data.append(
            #     [player[0], player[1], food[0], food[1], action, str(was_random), reward, str(done), str(prediction)])
            target_action_score = reward + discount_rate * np.max(model.predict(
                np.array([new_observation.reshape(5, 5, 1)])))

            label = prediction
            label[0][action] = target_action_score
            model.fit(model_input, label, epochs=1,
                      verbose=0, callbacks=[WandbCallback()])

            print_step_after_move(reward, target_action_score,
                                  label, model.predict(model_input))

            if (reward > 0):
                total_food += 1
            step += 1

            observation = new_observation
        wandb.log({'episode': episode, 'total_food': total_food, 'eps': eps})
        print('Score: {}'.format(total_food))
        print()
    # wandb_table_data = []
    # for px in range(4):
    #     for py in range(4):
    #         for fx in range(4):
    #             for fy in range(4):
    #                 wandb_table_data.append([px, py, fx, fy, str(
    #                     model.predict(np.array([[px, py, fx, fy]])))])
    # wandb.log({'predictions': wandb.Table(rows=wandb_table_data,
    #                                       columns=['px', 'py', 'fx', 'fy', 'prediction'])})
    env.close()


env = gym.make('snake-v0', board_size=wandb.config.board_size,
               food_reward=wandb.config.food_reward, death_reward=wandb.config.death_reward)

model = keras.Sequential()
# model.add(keras.layers.InputLayer(
# batch_input_shape = (1, wandb.config.board_size**2)))
model.add(keras.layers.Conv2D(4,
                              (2, 2),
                              input_shape=(wandb.config.board_size,
                                           wandb.config.board_size, 1),
                              activation='relu'))
# model.add(keras.layers.MaxPool2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(40, activation='relu'))
model.add(keras.layers.Dense(40, activation='relu'))
model.add(keras.layers.Dense(4, activation='linear'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam())

model.summary()

train(env, model, num_episodes=1000000)
