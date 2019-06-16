import time
from debug import colors

debug = False
no_render = False
auto_play = False
fast_forward_remaining = 0

def render_env_until_key_press(env):
    if env.unwrapped.IN_COLAB:
        return

    global fast_forward_remaining, auto_play
    waiting = True

    def key_press(key, mod):
        nonlocal waiting
        global no_render, fast_forward_remaining, debug, auto_play

        # print('key', key, 'mod', mod)

        if (key == 115 and mod == 64): # cmd + s
            file_name = 'last_model.h5'
            model.save(file_name)
            print(colors.OKBLUE +
                  'Saved model to {}'.format(file_name) + colors.ENDC)

        if (key == 112):  # p
            auto_play = not auto_play
            print(colors.OKBLUE +
                  'Turned {} autoplay'.format('on' if auto_play else 'off') + colors.ENDC)

        if (key == 100):  # d
            debug = not debug
            print(colors.OKBLUE +
                  'Turned {} debug'.format('on' if debug else 'off') + colors.ENDC)
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
            print(colors.OKBLUE + "Fast forwarding {} steps {}".format(
                fast_forward_remaining + 1, '(no render)' if no_render else '') + colors.ENDC)

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