import gym
import gym_snake
import time
import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np


wandb.init(project="snake")
wandb.config.discount_rate = .8
wandb.config.eps = .5
wandb.config.decay_factor = .999

no_render = False


def render_env_until_key_press(env):
    waiting = True

    def key_press(key, mod):
        nonlocal waiting
        global no_render
        if (key == 99):  # c
            no_render = True
        waiting = False

    if not no_render:
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        while waiting:
            time.sleep(0.01)
            env.render()
            pass


def print_step_before_move(step, player, food, prediction, action, was_random):
    print()
    print('-' * 40, step, '-' * 40)
    print('Player is at             ', player)
    print('Food is at               ', food)
    print('Prediction:              ', prediction)

    action_str = '{} ({})'.format(
        action, ['up', 'right', 'down', 'left'][action])
    if was_random:
        print('Random move:             ', action_str)
    else:
        print('Move:                    ', action_str)


def print_step_after_move(reward, target_action_score, label, new_prediction):
    print()
    print('Reward:                  ', reward)
    print('Target action score:     ', target_action_score)
    print('Label:                   ', label)
    print('New prediction:          ', new_prediction)
    print('-' * 83)
    print()


def train(env, model, num_episodes=500):
    discount_rate = wandb.config.discount_rate
    eps = wandb.config.eps
    decay_factor = wandb.config.decay_factor
    # counter = []
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        player, food = env.reset()
        eps *= decay_factor
        done = False
        total_food = 0
        step = 0
        # wandb_table_data = []
        while not done:
            model_input = np.array([[player[0], player[1], food[0], food[1]]])
            prediction = model.predict(model_input)
            if np.random.random() < eps:
                action = np.random.randint(0, 4)
                was_random = True
            else:
                action = np.argmax(prediction)
                was_random = False
            # counter.append(4**4 * action + 4**3 * player[0] + 4**2 * player[1] +
            #                4 * food[0] + food[1])
            # print_step_before_move(
            #     step, player, food, prediction, action, was_random)

            if episode > 2000:
                render_env_until_key_press(env)
            observation, reward, done, _ = env.step(action)
            # wandb_table_data.append(
            #     [player[0], player[1], food[0], food[1], action, str(was_random), reward, str(done), str(prediction)])
            new_player, new_food = observation
            target_action_score = reward + discount_rate * np.max(model.predict(
                np.array([[new_player[0], new_player[1], food[0], food[1]]])))

            label = prediction
            label[0][action] = target_action_score
            model.fit(model_input, label, epochs=1, verbose=0)

            # print_step_after_move(reward, target_action_score,
            #                       label, model.predict(model_input))

            if (reward > 0):
                total_food += 1
            step += 1

            player, food = new_player, new_food
        wandb.log({'total_food': total_food})
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


env = gym.make('snake-v0')

model = keras.Sequential()
model.add(keras.layers.InputLayer(batch_input_shape=(1, 4)))
# model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(4, activation='linear'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam())

train(env, model, num_episodes=10000)
