import gym
import gym_snake
import wandb
import algorithms.basic
import algorithms.eps_grower
import algorithms.replayer
import algorithms.rnd
import algorithms.not_epsilon
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import keras
from debug import colors, debugger
import numpy as np

wandb.init(project="snake")
conf = wandb.config
conf.discount_rate = .95
conf.initial_eps = 1
conf.decay_factor = .999
conf.min_eps = .005
conf.board_size = 9
conf.food_reward = 1
conf.death_reward = -1
conf.super_random_training = False
conf.easy_world = False
conf.load_model = 'last_model.h5'
# conf.load_model = None
conf.algorithm = 'inference'


env = gym.make('snake-v0', 
               board_size=conf.board_size,
               food_reward=conf.food_reward, 
               death_reward=conf.death_reward, 
               super_random_training=conf.super_random_training,
               easy_world=conf.easy_world)

if conf.load_model is not None:
    model = keras.models.load_model(conf.load_model)
    print(colors.OKBLUE +
        'Loaded model from {}'.format(conf.load_model) + colors.ENDC)
else:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(4,
                                (2, 2),
                                input_shape=(conf.board_size,
                                            conf.board_size, 3),
                                padding='same',
                                activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(8, activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4, activation='linear'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())

model.summary()

if conf.algorithm == 'basic':
    algorithms.basic.train(conf, env, model, num_episodes=1000000)
elif conf.algorithm == 'replayer':
    algorithms.replayer.train(conf, env, model, num_episodes=1000000)
elif conf.algorithm == 'rnd':
    algorithms.rnd.train(conf, env, model, num_episodes=1000000)
elif conf.algorithm == 'not_epsilon':
    algorithms.not_epsilon.train(conf, env, model, num_episodes=1000000)
elif conf.algorithm == 'eps_grower':
    algorithms.eps_grower.train(conf, env, model, num_episodes=1000000)
elif conf.algorithm == 'baseline':
    # model = DQN(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=1000000)
    # model.save("mlp_baseline")

    # del model # remove to demonstrate saving and loading

    model = DQN.load("mlp_baseline")

    while True:
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, state = model.predict(observation)
            observation, reward, done, info = env.step(action)
            if reward > 0:
                score += 1
            debugger.render_env_until_key_press(env, None)
        print('Score:', score)
        
elif conf.algorithm == 'inference':
    while True:
        observation = env.reset()
        done = False
        total_food = 0
        while not done:
            model_input = np.array([observation])
            prediction = model.predict(model_input)
            action = np.argmax(prediction)

            debugger.render_env_until_key_press(env, None)

            new_observation, reward, done, _ = env.step(action)
            if reward > 0:
                total_food += 1

            target_action_score = reward + (0 if done else conf.discount_rate * np.max(model.predict(
                np.array([new_observation]))))

            label = prediction
            label[0][action] = target_action_score
            model.fit(model_input, label, epochs=1,
                      verbose=0)

            observation = new_observation            


        wandb.log({'total_food': total_food})
        print('Score:', total_food)