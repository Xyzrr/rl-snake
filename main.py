import gym
import gym_snake
import wandb
import algorithms.basic
import algorithms.replay
import keras
from debug import colors

wandb.init(project="snake")
conf = wandb.config
conf.discount_rate = .9
conf.initial_eps = 0
conf.eps_growth_rate = 1.01
conf.decay_factor = .999
conf.board_size = 9
conf.food_reward = 1
conf.death_reward = -1
conf.experience_replay = False
# conf.load_model = None
conf.load_model = 'last_model.h5'
conf.algorithm = 'basic'


env = gym.make('snake-v0', board_size=conf.board_size,
               food_reward=conf.food_reward, death_reward=conf.death_reward)


if conf.load_model is not None:
    model = keras.models.load_model(conf.load_model)
    print(colors.OKBLUE +
        'Loaded model from {}'.format(conf.load_model) + colors.ENDC)
else:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(8,
                                (2, 2),
                                input_shape=(conf.board_size,
                                            conf.board_size, 4),
                                activation='relu'))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(100, activation='relu'))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4, activation='linear'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())


model.summary()


if conf.algorithm == 'basic':
    algorithms.basic.train(conf, env, model, num_episodes=1000000)
elif conf.algorithm == 'replay':
    algorithms.replay.train(conf, env, model, num_episodes=1000000)