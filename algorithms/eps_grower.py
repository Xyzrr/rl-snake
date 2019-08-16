import gym
import wandb
from wandb.keras import WandbCallback
import numpy as np
from debug import colors, debugger


def train(conf, env, model, num_episodes=500, eps_growth_rate=1.005):
    discount_rate = conf.discount_rate
    eps = conf.initial_eps
    conf.eps_growth_rate = eps_growth_rate
    decay_factor = conf.decay_factor
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        observation = env.reset()
        eps = max(eps*decay_factor, conf.min_eps)
        done = False
        total_food = 0
        step = 0
        while not done:
            model_input = np.array(
                [observation])
            prediction = model.predict(model_input)

            adjusted_eps = eps * eps_growth_rate ** step
            if np.random.random() < adjusted_eps:
                action = np.random.randint(0, 4)
                was_random = True
            else:
                action = np.argmax(prediction)
                was_random = False

            debugger.print_step_before_move({
                'step': step,
                'observation': observation,
                'prediction': prediction,
                'was_random': was_random,
                'action': action,
            })


            debugger.render_env_until_key_press(env, model)

            new_observation, reward, done, _ = env.step(action)

            target_action_score = reward + (0 if done else discount_rate * np.max(model.predict(
                np.array([new_observation]))))

            label = prediction
            label[0][action] = target_action_score
            model.fit(model_input, label, epochs=1,
                      verbose=0)

            debugger.print_step_after_move({'reward': reward, 
                'target_action_score': target_action_score,
                'label': label, 
                'new_prediction': model.predict(model_input)
            })

            if (reward > 0):
                total_food += 1
            step += 1

            observation = new_observation
        wandb.log({'episode': episode, 'total_food': total_food,
                   'eps': eps, 'lifetime': step, 'adjusted_eps': adjusted_eps})
        print('Score: {}'.format(total_food))
        print()
    env.close()