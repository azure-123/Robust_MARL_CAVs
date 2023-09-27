"""Self Play
"""

import argparse
import os
import numpy as np
import torch
import highway_env
import gym
import configparser
from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.senario_battle import play

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps
def piecewise_decay(now_step, anchor, anchor_value):
    """piecewise linear decay scheduler
    Parameters
    ---------
    now_step : int
        current step
    anchor : list of integer
        step anchor
    anchor_value: list of float
        value at corresponding anchor
    """
    i = 0
    while i < len(anchor) and now_step >= anchor[i]:
        i += 1

    if i == len(anchor):
        return anchor_value[-1]
    else:
        return anchor_value[i-1] + (now_step - anchor[i-1]) * \
                                   ((anchor_value[i] - anchor_value[i-1]) / (anchor[i] - anchor[i-1]))
# def linear_decay(now_step, total_step, final_value):
#     """linear decay scheduler"""
#     decay = (1 - final_value) / total_step
#     return max(final_value, 1 - decay * now_step)
def linear_increase(now_step,total_step,final_value):
    increase = final_value / total_step
    return min(final_value,increase*now_step)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'dqn'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=100, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=20000, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--max_steps', type=int, default=100, help='set the max steps')


    args = parser.parse_args()
    config_dir = './examples/battle_model/config/highway.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    # Initialize the environment
    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getfloat('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getfloat('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getfloat('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')

    assert env.T % args.max_steps == 0
    # env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    # handles = env.get_handles()

    log_dir = os.path.join(BASE_DIR,'data/tb_logs')
    model_dir = os.path.join(BASE_DIR, 'data/models/new/3{}'.format(args.algo))
    algo_model_dir = os.path.join(BASE_DIR, 'data/models/3-0-safemfac/')
    models = spawn_ai(args.algo, env,  args.algo + '-me', args.max_steps)
    # models.load(algo_model_dir,20000)

    runner = tools.Runner(env, args.max_steps, models, play, render=args.render, save_every=args.save_every, tau=0.01, log_name=args.algo,
        log_dir=log_dir, model_dir=model_dir, train=True)

    for k in range(0, args.n_round):
        eps = piecewise_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        # if k < int(args.n_round * 0.01):
        #     eps_attack = 0
        # else:
        # if k<2000:
        #     eps_attack = 0
        # elif k > 4800:
        #     eps_attack = 0.7
        # else:
        # if k < int(args.n_round * 0.2):
        #     eps_attack = 0
        # else:
        #     eps_attack = linear_increase(k-int(args.n_round * 0.2) ,int(args.n_round*0.8)-int(args.n_round * 0.2),0.2)# 0.175
        # eps_attack = linear_increase(k ,int(args.n_round*0.7),0.5) 
        eps_attack = 0
        runner.run(eps, k, eps_attack)