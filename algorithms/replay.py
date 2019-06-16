import gym
import wandb
from wandb.keras import WandbCallback
import numpy as np
from debug import colors, debugger


# def fit_model_to_memory(model, replay_memory, minibatch_size=30):
#     model_input = np.stack([observation.reshape(
#         conf.board_size, conf.board_size, 1) for observation, _, _, _ in replay_memory[-30:]])
#     predictions = model.predict(model_input)


def train(conf, env, model, num_episodes=500):
    discount_rate = conf.discount_rate
    eps = conf.initial_eps
    decay_factor = conf.decay_factor
    replay_memory = []
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        observation = env.reset()
        eps *= decay_factor
        done = False
        total_food = 0
        step = 0
        while not done:
            model_input = np.array(
                [observation.reshape(conf.board_size, conf.board_size, 4)])
            prediction = model.predict(model_input)
            adjusted_eps = eps * conf.eps_growth_rate ** step
            if np.random.random() < adjusted_eps:
                action = np.random.randint(0, 4)
                was_random = True
            else:
                action = np.argmax(prediction)
                was_random = False

            debugger.print_step_before_move(
                step, observation, prediction, action, was_random)

            debugger.render_env_until_key_press(env)

            new_observation, reward, done, _ = env.step(action)
            memory_unit = (observation, action, reward, new_observation)
            replay_memory.append(memory_unit)

            # fit_model_to_memory(model, replay_memory)
            target_action_score = reward + (0 if done else discount_rate * np.max(model.predict(
                np.array([new_observation.reshape(conf.board_size, conf.board_size, 4)]))))

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
        wandb.log({'episode': episode, 'total_food': total_food,
                   'eps': eps, 'lifetime': step, 'adjusted_eps': adjusted_eps})
        print('Score: {}'.format(total_food))
        print()
    env.close()