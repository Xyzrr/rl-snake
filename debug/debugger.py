import time
from debug import colors

debug = False
no_render = False
auto_play = False
fast_forward_remaining = 0

def render_env_until_key_press(env, model):
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

def observation_to_string(observation):
    result = ''
    for i, row in enumerate(observation):
        line = ''
        for j, cell in enumerate(row):
            if cell[0]:
                line += colors.FAIL + "O" + colors.ENDC
            elif cell[1]:
                line += colors.OKGREEN + "O" + colors.ENDC
            elif cell[2]:
                line += colors.OKGREEN + (str(int(cell[2])) if cell[2] < 10 else chr(int(cell[2]) - 10 + ord('a'))) + colors.ENDC
            else:
                line += "."
        result += line + "\n"
    return result

def print_step_before_move(data):
    if not debug:
        return

    print()
    print('-' * 40, data['step'], '-' * 40)
    if 'observation' in data:
        print('Observation:             ')
        print(observation_to_string(data['observation']))
    if 'prediction' in data:
        print('Prediction:              ', data['prediction'])
    if 'randomized_prediction' in data:
        print('Randomized:              ', data['randomized_prediction'])

    if 'action' in data:
        action_str = '{} ({})'.format(
            data['action'], ['up', 'right', 'down', 'left'][data['action']])
        if 'greedy_action' in data and data['greedy_action'] != data['action']:
            greedy_action_str = '{} ({})'.format(
                data['greedy_action'], ['up', 'right', 'down', 'left'][data['greedy_action']])
            print('Move:                    ', action_str, colors.WARNING + 'greedy:', greedy_action_str + colors.ENDC)
        else:
            print('Move:                    ', action_str)

        if 'was_random' in data and data['was_random']:
            print('Was random!')


def print_step_after_move(data):
    if not debug:
        return

    print()
    if 'reward' in data:
        print('Reward:                  ', data['reward'])
    if 'target_action_score' in data:
        print('Target action score:     ', data['target_action_score'])
    if 'label' in data:
        print('Label:                   ', data['label'])
    if 'new_prediction' in data:
        print('New prediction:          ', data['new_prediction'])
    print('-' * 83)
    print()