from util.replay_buffer import ReplayBuffer
import gym
import wandb
from wandb.keras import WandbCallback
import numpy as np
from debug import colors, debugger


def train(conf, env, model, num_episodes=500, batch_size=100, buffer_size=10000):
    conf.buffer_size = buffer_size
    conf.batch_size = batch_size

    replay_buffer = ReplayBuffer(size=buffer_size)
    discount_rate = conf.discount_rate
    eps = conf.initial_eps
    decay_factor = conf.decay_factor
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        observation = env.reset()
        eps *= decay_factor
        done = False
        total_food = 0
        step = 0
        while not done:
            model_input = np.array(
                [observation])
            prediction = model.predict(model_input)
            if np.random.random() < eps:
                action = np.random.randint(0, 4)
                was_random = True
            else:
                action = np.argmax(prediction)
                was_random = False

            debugger.print_step_before_move(
                step, observation, prediction, action, was_random)

            debugger.render_env_until_key_press(env)

            new_observation, reward, done, _ = env.step(action)

            replay_buffer.add(observation, action, reward, new_observation, float(done))

            # target_action_score = reward + (0 if done else discount_rate * np.max(model.predict(
            #     np.array([new_observation]))))

            # label = prediction
            # label[0][action] = target_action_score
            # model.fit(model_input, label, epochs=1,
            #           verbose=0)

            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            labels = model.predict(obses_t)
            targets = discount_rate * np.max(model.predict(obses_tp1), axis=1)
            # print('targets', targets)
            # print('rewards', rewards)
            for i in range(len(dones)):
                if dones[i]:
                    targets[i] = 0
                targets[i] += rewards[i]
                labels[i][actions[i]] = targets[i]
            model.fit(obses_t, labels, epochs=1, verbose=0)

            weights, batch_idxes = np.ones_like(rewards), None

            # debugger.print_step_after_move(reward, target_action_score,
            #                       label, model.predict(model_input))

            if (reward > 0):
                total_food += 1
            step += 1

            observation = new_observation
        wandb.log({'episode': episode, 'total_food': total_food,
                   'eps': eps, 'lifetime': step})
        print('Score: {}'.format(total_food))
        print()
    env.close()