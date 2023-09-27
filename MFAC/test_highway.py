"""Battle
"""

import argparse
import os
import numpy as np
import configparser
import gym
import highway_env
from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.senario_battle import test


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'dqn'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--n_round', type=int, default=1, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--max_steps', type=int, default=100, help='set the max steps')
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    # parser.add_argument('--idx', nargs='*', required=True)


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

    algo_model_dir = os.path.join(BASE_DIR, 'data/models/new/2mfac/')
    models = spawn_ai(args.algo, env,  args.algo + '-me', args.max_steps)
    models.load(algo_model_dir,20000)
    total_reward=[] 
    avg_speed=[]
    crash_ra=[]
    seeds = [int(s) for s in args.evaluation_seeds.split(',')]
    print(seeds)
    for i in range(len(seeds)):
        print(i,":",seeds[i])
        env.config['seed'] = seeds[i]
        for k in range(0, args.n_round):
            runner = tools.Runner(env, args.max_steps, models, test, render=args.render, tau=0.01, 
            train=False)

            runner.run(0, k, 1, total_reward=total_reward , avg_speed=avg_speed, crash_ra=crash_ra)

    print('Ave-Reward:', np.mean(total_reward), 'Ave-Speed:',np.mean(avg_speed), 'Crashs:', np.mean(crash_ra))