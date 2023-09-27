import random
import math
import numpy as np
from attacks import attack

def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    env.add_agents(handles[0], method="random", n=map_size * map_size * 0.02)
    env.add_agents(handles[1], method="random", n=map_size * map_size * 0.04)

def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False,eps_attack=0):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    # print(eps_attack)
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums),'esp_atk:',eps_attack)
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
        for i in range(n_group):
            
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        buffer1 = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer1['prob'] = former_act_prob[1]

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        if train:
            models[0].flush_buffer(**buffer)
            models[1].flush_buffer(**buffer1)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()
            
        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train(eps=eps,eps_attack=eps_attack)
        models[1].train(eps=eps,eps_attack=eps_attack)


    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
    
def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False,eps_attack=0):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)
    attacked = False
    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]
    # attack_agent_id = random.randint(0,63)
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
        # print(ids[0])
        if attacked:
            state1=[]
            state2=[]
            former_act=[]
            start=0
            num_attack=64
            for attack_agent_id in range(start,start+num_attack):
                if ids[0][0]>63:
                    attack_agent_id += 64
                if attack_agent_id in ids[0] :
                    attack_agent_index = ids[0].tolist().index(attack_agent_id)
                    state1.append(state[0][0][attack_agent_index])
                    state2.append(state[0][1][attack_agent_index])
                    former_act.append(former_act_prob[0][attack_agent_index])
            if len(state2)>0:
                attack_config={"params": {"niters": 10},'method':'pgd','epsilon':'0.1'}
                adv_state = attack(models[0], [np.array(state1),np.array(state2)],np.array(former_act),eps, attack_config)
                i = 0
                for attack_agent_id in range(start,start+num_attack):
                    if ids[0][0]>63:
                        attack_agent_id += 64
                    if attack_agent_id in ids[0] :
                        attack_agent_index= ids[0].tolist().index(attack_agent_id)
                        # print(attack_agent_id,attack_agent_index)
                        state[0][0][attack_agent_index]=adv_state[0][i]
                        state[0][1][attack_agent_index]=adv_state[1][i]
                        i += 1
            # attack_config={"params": {"niters": 10}}
            # state[0] = attack(models[0], state[0],former_act_prob[0],eps, attack_config)
        for i in range(n_group):
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)


        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1
        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
