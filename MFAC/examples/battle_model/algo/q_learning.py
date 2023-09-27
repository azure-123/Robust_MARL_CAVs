import os
import torch
import numpy as np

from . import base
from . import tools

class DQN(base.ValueNet):
    """docstring for DQN"""
    def __init__(self, handle, env, sub_len, eps=1.0, memory_size=2**10, batch_size=32):
        super().__init__(env, handle)
        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self,eps):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feats, obs_next, feat_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=[obs, feats], target_q=target_q, acts=actions, masks=masks,eps=eps)
            self.update()
            # if i % 50 == 0:
            #     print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path,step=0):
        file_path = os.path.join(dir_path, "dqn"+str(step)+".pth")
        torch.save(self.Q.state_dict(), file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path,step=0):
        file_path = os.path.join(dir_path, "dqn.pth")
        self.Q.load_state_dict(torch.load(file_path))
        print("[*] Loaded model from {}".format(file_path))


class MFQ(base.ValueNet):
    def __init__(self, handle, env, sub_len, eps=1.0, memory_size=2**10, batch_size=512):
        super().__init__(env, handle, use_mf=True)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.replay_buffer = tools.MemoryGroup(**config)

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self,eps,eps_attack):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()

            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts, masks=masks,eps=eps,eps_attack=eps_attack)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:',loss, '/ Q:', q)

    def save(self, dir_path,step=0):
        file_path = os.path.join(dir_path, "mfq"+str(step)+".pth")
        torch.save(self.Q.state_dict(), file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path,step=0):
        file_path = os.path.join(dir_path, "mfq"+str(step)+".pth")
        self.Q.load_state_dict(torch.load(file_path))
        print("[*] Loaded model from {}".format(file_path))

