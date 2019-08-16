import gym
import wandb
from wandb.keras import WandbCallback
import numpy as np
from debug import colors, debugger
import keras


class RND:
    def __init__(self, conf):
        self.conf = conf
        self.fixed_model = self.build_model()
        self.mimic_model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(8,
                                    (2, 2),
                                    input_shape=(self.conf.board_size,
                                                self.conf.board_size, 3),
                                    activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        return model

    def observe(self, observation):
        label = self.fixed_model.predict(np.array([observation]))
        hist = self.mimic_model.fit(np.array([observation]), label, epochs=1, verbose=0)
        print('HISTORY:', hist.history)
        return hist.history['loss'][0]


def train(conf, env, model, num_episodes=500):
    discount_rate = conf.discount_rate
    decay_factor = conf.decay_factor
    rnd = RND(conf)
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        observation = env.reset()
        done = False
        total_food = 0
        step = 0
        while not done:
            model_input = np.array([observation])
            prediction = model.predict(model_input)
            action = np.argmax(prediction)

            debugger.print_step_before_move(
                step, observation, prediction, action, False)

            debugger.render_env_until_key_press(env)

            new_observation, reward, done, _ = env.step(action)

            if not done:
                reward += 10*rnd.observe(new_observation)

            target_action_score = reward + (0 if done else discount_rate * np.max(model.predict(
                np.array([new_observation]))))

            label = prediction
            label[0][action] = target_action_score
            model.fit(model_input, label, epochs=1,
                      verbose=0)

            debugger.print_step_after_move(reward, target_action_score,
                                  label, model.predict(model_input))

            if (reward > 0):
                total_food += 1
            step += 1

            observation = new_observation
        # wandb.log({'episode': episode, 'total_food': total_food,
        #            'lifetime': step})
        print('Score: {}'.format(total_food))
        print()
    env.close()