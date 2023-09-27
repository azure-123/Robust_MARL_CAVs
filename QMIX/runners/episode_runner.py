# from envs import REGISTRY as env_REGISTRY
import gym
import os
import highway_env
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from adv.attack import attack_gd
from adv.attack_de import attack_de
from adv.attack_q_de import attack_q_de
from adv.attack_target import attack_target
from utils1.utils import index_to_one_hot, to_tensor_var, VideoRecorder
import random
from learners import REGISTRY as le_REGISTRY
import torch

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.adv_batch_size = self.args.adv_batch_size_run
        assert self.batch_size == 1

        self.env = gym.make(args.env)
        self.env.config['seed'] = args.env_args['seed']
        self.env.config['simulation_frequency'] = args.env_args['simulation_frequency']
        self.env.config['duration'] = args.env_args['duration']
        self.env.config['policy_frequency'] = args.env_args['policy_frequency']
        self.env.config['COLLISION_REWARD'] = args.env_args['COLLISION_REWARD']
        self.env.config['HIGH_SPEED_REWARD'] = args.env_args['HIGH_SPEED_REWARD']
        self.env.config['HEADWAY_COST'] = args.env_args['HEADWAY_COST']
        self.env.config['HEADWAY_TIME'] = args.env_args['HEADWAY_TIME']
        self.env.config['MERGING_LANE_COST'] = args.env_args['MERGING_LANE_COST']
        self.env.config['traffic_density'] = args.env_args['traffic_density']
        self.env.config['safety_guarantee'] = args.env_args['safety_guarantee']
        # traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
        # print(args.env_args['traffic_density'])
        
        self.env.config['action_masking'] = args.env_args['action_masking']
        self.env.config['n_step'] = args.env_args['n_step']

        self.episode_limit = args.env_args['roll_out_n_steps']
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def setup_adv(self, scheme, groups, preprocess, mac, adv_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.adv_mac = adv_mac
    
    def get_avail_actions(self):
        return np.ones((self.n_agent, self.env.n_a))

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.adv_batch = self.new_batch()
        self.adv_opp_batch = self.new_batch()
        # self.env.reset()
        self.env_state, _ = self.env.reset()
        self.n_agent = len(self.env.controlled_vehicles)
        self.t = 0

        # env_info = {"state_shape": self.n_agent*self.env.n_s,
        #             "obs_shape": self.env.n_s,
        #             "n_actions": self.env.n_a,
        #             "n_agents": self.n_agent,
        #             "episode_limit": self.episode_limit}
        # return env_info
    
    def get_env_info(self):
        self.env.reset()
        self.n_agent = len(self.env.controlled_vehicles)

        env_info = {"state_shape": self.n_agent*self.env.n_s,
                    "obs_shape": self.env.n_s,
                    "n_actions": self.env.n_a,
                    "n_agents": self.n_agent,
                    "episode_limit":self.episode_limit}
        return env_info


    def run(self, test_mode=False, learner = None, adv_test = False):
        env_info_x = self.get_env_info()
        self.reset()

        terminated = False
        episode_return = 0
        self.hidden_state = self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.Number_attack > 0 and self.args.attack_method == "adv_tar":
            self.adv_hidden_state = self.adv_mac.init_hidden(batch_size=self.batch_size)
        # env_info = self.get_env_info()
        obs_shape = env_info_x["obs_shape"]
        n_agents = env_info_x["n_agents"]
        # print(n_agents)
        # aaa
        n_actions = env_info_x["n_actions"]
        
        if test_mode and self.args.video_filename is not None:
            video_filename = os.path.join(self.args.video_filename,
                                          "testing_episode{}".format(self.t_env + 1) + '_{}'.format(test_mode) +
                                          '.avi')
            rendered_frame = self.env.render(mode="rgb_array")
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,5))
            video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)
            video_recorder.add_frame(rendered_frame)
        else:
            video_recorder = None

        while not terminated:

            pre_transition_data = {
                "state": [np.concatenate(self.env_state)],
                "avail_actions": [self.get_avail_actions()],
                "obs": [self.env_state],
            }
            # print(pre_transition_data)

            # ad_obs 需要有两个batch normal_batch adv_batch
            # print(pre_transition_data)
            # print(self.t)
            self.batch.update(pre_transition_data, ts=self.t)    
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.Number_attack > 0 and adv_test and (self.args.attack_method == "adv_de" or self.args.attack_method == "q_adv_de"):
                actions, hidden_state_true, action_values = self.mac.action_value(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
            else:
                actions, hidden_state_true = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)

            ###############################################################

            if self.args.Number_attack > 0 and adv_test:
                if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd" or self.args.attack_method == "rand_nosie":
                    adv_inputs_obs = attack_gd(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    # attacked_agent_id =  np.random.randint(0,high = n_agents, size = self.args.Number_attack)
                    attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    # print(attacked_agent_id)
                    # print(adv_inputs)
                    # print(adv_inputs_obs)
                    for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]][:obs_shape].copy()
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_inputs)
                    # print("*****************************************************")
                    # print(adv_transition_data["obs"])
                    self.adv_batch.update(adv_transition_data, ts=self.t)
                    actions, hidden_state_true = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                    self.env_state, reward, terminated, env_info = self.env.step(tuple(actions[0].cpu().data.numpy()))
                    episode_return += reward

                    post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated,)],
                    }
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_true
                    self.adv_batch.update(post_transition_data, ts=self.t)

                elif self.args.attack_method == "adv_tar":
                    tar_actions, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    ################*********************##################*******************
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(pre_transition_data["obs"][0])
                    # print("**********************************")
                    # print(adv_inputs)    
                    # attacked_agent_id =  np.random.randint(0,high = n_agents, size = self.args.Number_attack)
                    attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    # print(attacked_agent_id)
                    for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                        # adv_inputs[i] = adv_inputs_obs[i].copy()
                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                        # "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    # print(adv_transition_data["obs"])
                    # print("****************************************")
                    # print(pre_transition_data["obs"][0])

                    self.adv_batch.update(adv_transition_data, ts=self.t)
                    self.adv_opp_batch.update(pre_transition_data, ts=self.t) # 攻击者输入正常状态
                    
                    adv_actions, hidden_state_= self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    # print(adv_actions - tar_actions)
                    # print()
                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    opp_post_transition_data = {
                        "actions": tar_actions,
                        "reward": [(-reward,)], # 可做修改
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t) 
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                    self.adv_hidden_state = adv_hidden_state_
                ################################################################
                elif self.args.attack_method == "adv_de":
                    
                    chosen_action_qvals = torch.gather(action_values[0], dim=1, index=np.transpose(actions))#.unsqueeze(1)
                    size_0 = chosen_action_qvals.size(0)
                    chosen_action_qvals = chosen_action_qvals.view(1, size_0)
                    if self.args.mixer == "vdn":
                        chosen_action_qvals = chosen_action_qvals.unsqueeze(0)
                    # print(chosen_action_qvals)
                    # aaaa
                    mix_value_true = learner.mix_value(chosen_action_qvals, torch.FloatTensor(pre_transition_data["state"][0]))
                    # print(mix_value_true)
                    # aaa
                    tar_actions, xx = attack_de(img = actions[0], label = mix_value_true, learner = learner, action_values=action_values,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= pre_transition_data["state"],\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False)# state, qimx, actions, qmix_value_true, avilable_actions
                    
                   
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                 
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    for i in range (self.args.Number_attack):
                        # print(adv_inputs[0])
                        adv_inputs[xx[2*i]] = adv_inputs_obs[xx[2*i]]
                    # print(adv_inputs)
                    # print(obs_shape)
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                    }
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    adv_actions, hidden_state_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_

                elif self.args.attack_method == "q_adv_de":
                    
                    # chosen_action_qvals = torch.gather(action_values[0], dim=1, index=np.transpose(actions))#.unsqueeze(1)
                    # size_0 = chosen_action_qvals.size(0)
                    # chosen_action_qvals = chosen_action_qvals.view(1, size_0)
                    # print(actions)
                    # print(pre_transition_data["state"][0])

                    mix_value_true = learner.mix_value(actions, torch.FloatTensor(pre_transition_data["state"]))
                    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    tar_actions, xx = attack_q_de(img = actions[0], label = mix_value_true, learner = learner, actions=actions,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= pre_transition_data["state"],\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False)# state, qimx, actions, qmix_value_true, avilable_actions
                    
                   
                    adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                 
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)    
                    for i in range (self.args.Number_attack):
                        # print(adv_inputs[0])
                        adv_inputs[xx[2*i]] = adv_inputs_obs[xx[2*i]]
                    # print(adv_inputs)
                    # print(obs_shape)
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                    }
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    adv_actions, hidden_state_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated,)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
    
            else:
                # obs, reward, done, info
                # print(actions)
                self.env_state, reward, terminated, env_info = self.env.step(tuple(actions[0].cpu().data.numpy()))
                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated,)],
                }
                self.batch.update(post_transition_data, ts=self.t)
                self.hidden_state = hidden_state_true

                
                if video_recorder is not None:
                    rendered_frame = self.env.render(mode="rgb_array")
                    video_recorder.add_frame(rendered_frame)
                    # video_recorder.release()
                    # self.env.close()
                
            self.t += 1
        # pre_transition_data = {
        #         "state": [np.concatenate(self.env_state)],
        #         "avail_actions": [self.get_avail_actions()],
        #         "obs": [self.env_state]
        #     }
        
        last_data = {
            "state": [np.concatenate(self.env_state)],
            "avail_actions": [self.get_avail_actions()],
            "obs": [self.env_state]
        }
        self.batch.update(last_data, ts=self.t)
        if self.args.Number_attack > 0 and adv_test and (self.args.attack_method == "adv_de" or self.args.attack_method == "q_adv_de"):
            actions, hid, action_values = self.mac.action_value(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
        else:
            actions, hid = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
        # print(actions)
        self.batch.update({"actions": actions}, ts=self.t)
        if self.args.Number_attack > 0 and adv_test:
            if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd" or self.args.attack_method == "rand_nosie":
                adv_inputs_obs = attack_gd(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, hidden_states=self.hidden_state)
                adv_inputs = pre_transition_data["obs"][0].copy()
                # print(adv_inputs)    
                # attacked_agent_id =  np.random.randint(0,high = n_agents, size = self.args.Number_attack)
                attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)


                # print(attacked_agent_id)
                for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                    adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]][:obs_shape].copy()
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [arr[:obs_shape] for arr in adv_inputs]
                    # "obs": [adv_inputs[:,0:obs_shape]]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)

            elif self.args.attack_method == "adv_tar":
                tar_actions, hid = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, hidden_state=self.hidden_state)
                adv_inputs = pre_transition_data["obs"][0].copy()
                # print(adv_inputs)    
                # attacked_agent_id =  np.random.randint(0,high = n_agents, size = self.args.Number_attack)
                attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                for i in range (self.args.Number_attack):
                # print(adv_inputs[0])
                    # adv_inputs[i] = adv_inputs_obs[i].copy()
                    adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()
                # adv_transition_data = {
                #     "state": pre_transition_data["state"],
                #     "avail_actions": pre_transition_data["avail_actions"],
                #     "obs": [arr[:obs_shape] for arr in adv_inputs]
                #     # "obs": [adv_inputs[:,0:obs_shape]]
                # }
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [arr[:obs_shape] for arr in adv_inputs]
                    #"obs": [adv_inputs[:,0:obs_shape]]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)
                self.adv_opp_batch.update(last_data, ts=self.t)
                self.adv_opp_batch.update({"actions": tar_actions}, ts=self.t)

            elif self.args.attack_method == "adv_de" :

                chosen_action_qvals = torch.gather(action_values[0], dim=1, index=np.transpose(actions))#.unsqueeze(1)
                size_0 = chosen_action_qvals.size(0)
                chosen_action_qvals = chosen_action_qvals.view(1, size_0)
                if self.args.mixer == "vdn":
                        chosen_action_qvals = chosen_action_qvals.unsqueeze(0)
                mix_value_true = learner.mix_value(chosen_action_qvals, torch.FloatTensor(last_data["state"][0]))
                  
                tar_actions, xx = attack_de(img = actions[0], label = mix_value_true, learner = learner, action_values=action_values,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= last_data["state"],\
                                                    agents_available_actions = last_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False)# state, qimx, actions, qmix_value_true, avilable_actions
                    
                   
                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                 
                adv_inputs = pre_transition_data["obs"][0].copy()
                # print(adv_inputs)    
                for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                    adv_inputs[xx[2*i]] = adv_inputs_obs[xx[2*i]]

                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [arr[:obs_shape] for arr in adv_inputs]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)

            elif self.args.attack_method == "q_adv_de" :
                # print('xxxxxxxxxxxx')
                mix_value_true = learner.mix_value(actions, torch.FloatTensor(pre_transition_data["state"]))
                    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                tar_actions, xx = attack_q_de(img = actions[0], label = mix_value_true, learner = learner, actions=actions,\
                                                    n_agent = n_agents, n_action= n_actions, state_batch= pre_transition_data["state"],\
                                                    agents_available_actions = pre_transition_data["avail_actions"], pixels = self.args.Number_attack,\
                                                    maxiter=75, popsize=400, verbose=False)# state, qimx, actions, qmix_value_true, avilable_actions
                    # 75 400
                    # 75 500  0.5
                    # 100 500 0.38
                    # 200 500
                   
                   
                adv_inputs_obs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                 
                adv_inputs = pre_transition_data["obs"][0].copy()
                # print(adv_inputs)    
                for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                    adv_inputs[xx[2*i]] = adv_inputs_obs[xx[2*i]]

                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [arr[:obs_shape] for arr in adv_inputs]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)

        if video_recorder is not None:
            video_recorder.release()
        self.close_env()
        
        # Select actions in the last stored state  

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # print(env_info)
        # average_speed， crashed

        cur_stats.update({'average_speed': cur_stats.get('average_speed', 0) + env_info.get('average_speed', 0) })
        cur_stats.update({'crashed': cur_stats.get('crashed', 0) + (1 if np.sum(env_info.get('crashed', 0))>0 else 0) })
        # for k in set({'average_speed','crashed'})})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        # print(cur_stats["battle_won"])
        if self.args.evaluate:
            print(episode_return,'-------------', cur_stats["crashed"])
        if test_mode and (len(self.test_returns) == self.args.test_nepisode - 1):
            self._log(cur_returns, cur_stats, log_prefix)
            
            # self.logger.log_stat("episode", len(self.test_returns), self.t_env)

        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:

            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        if self.args.Number_attack > 0 and adv_test:
            if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd" or self.args.attack_method == "adv_de" or self.args.attack_method == "q_adv_de" or self.args.attack_method == "rand_nosie":
                return self.adv_batch
            elif self.args.attack_method == "adv_tar":
                return self.adv_batch, self.adv_opp_batch
        else:
            return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)

        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)

        stats.clear()
