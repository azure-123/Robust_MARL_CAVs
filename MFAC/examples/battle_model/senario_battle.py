import random
import math
import numpy as np
from .python.attacks import attack


def play(env, n_round, max_steps, models, print_every, eps=1.0, render=False, train=False,eps_attack=0):
    """play a ground and train"""
    state, _ = env.reset()
    n_agents = len(env.controlled_vehicles)
    n_action = env.n_a
    obs_shape = env.n_s
    step_ct = 0
    done = False


    # print(eps_attack)
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} ".format(n_round, eps),'esp_atk:',eps_attack)
    speeds = []
    rewards = []
    crashs = []
    former_act_prob = np.zeros((1, n_action))
    

    while not done and step_ct < max_steps:
        # take actions for every model

        former_act_prob = np.tile(former_act_prob, (n_agents, 1))

        acts = models.act(state=state, prob=former_act_prob, eps=eps)

        state, reward, done, env_info = env.step(tuple(acts))
        new_acts = env_info['new_action']
        # simulate one step

        buffer = {
            'state': state, 'acts': new_acts, 'rewards':reward
        }

        buffer['prob'] = former_act_prob
        buffer['terminal'] = done

        former_act_prob = np.mean(list(map(lambda x: np.eye(n_action)[x], new_acts)), axis=0, keepdims=True)

        if train:
            models.flush_buffer(**buffer)

        # stat info
        rewards.append(reward)
        speeds.append(env_info['average_speed'])
        crashs.append(env_info['crashed'])

        info = {"Ave-Reward": np.round(np.sum(rewards), decimals=6), 
                "Ave-Speed": np.round(np.mean(speeds), decimals=6),
                "Crashs": np.round(np.sum(crashs), decimals=1)}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models.train(eps=eps,eps_attack=eps_attack)

    total_rewards = np.sum(rewards)
    speed = np.mean(speeds)
    crash = np.sum(crashs)

    return total_rewards, speed, crash 
    
def test(env, n_round, max_steps, models, print_every, eps=1.0, render=False, train=False,eps_attack=0):
    """play a ground and train"""
    state, _ = env.reset(is_training=False, testing_seeds = env.config['seed'], num_CAV = 4)
    n_agents = len(env.controlled_vehicles)
    print(n_agents)
    n_action = env.n_a
    obs_shape = env.n_s
    step_ct = 0
    done = False
    attack_num = 0


    # print(eps_attack)
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} ".format(n_round, eps),'esp_atk:',eps_attack)
    speeds = []
    rewards = []
    crashs = []
    former_act_prob = np.zeros((1, n_action))
    

    while not done and step_ct < max_steps:
        # take actions for every model

        former_act_prob = np.tile(former_act_prob, (n_agents, 1))
        if attack_num>0:
            # model, X,former_act_prob,eps, attack_config,
            attack_config={"params": {"niters": 10,"epsilon":0.1},'method':'fgsm'}
            # print(state)
            state_adv = attack(models, state, former_act_prob, eps, attack_config)
            # print(state_adv)
            for i in range(attack_num):
                state[i] = state_adv[i].copy()
            # print(state)
            # aaaaaaaaa
        acts = models.act(state=state, prob=former_act_prob, eps=eps)

        state, reward, done, env_info = env.step(tuple(acts))
        # simulate one step

        former_act_prob = np.mean(list(map(lambda x: np.eye(n_action)[x], acts)), axis=0, keepdims=True)

        # stat info
        rewards.append(reward)
        speeds.append(env_info['average_speed'])
        crashs.append(env_info['crashed'])

        info = {"Ave-Reward": np.round(np.sum(rewards), decimals=6), 
                "Ave-Speed": np.round(np.mean(speeds), decimals=6),
                "Crashs": np.round(np.sum(crashs), decimals=1)}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))


    total_rewards = np.sum(rewards)
    speed = np.mean(speeds)
    crash = np.sum(crashs)

    return total_rewards, speed, crash 
