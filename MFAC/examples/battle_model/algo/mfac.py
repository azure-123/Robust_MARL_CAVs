import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import tools
from ..python.attacks import attack
from torch.autograd import Variable
# from ibp import network_bounds, base_network_bounds
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
def get_loglikelihood(self, p, actions):
    '''
    Inputs:
    - p, batch of probability tensors
    - actions, the actions taken
        '''
    try:
        dist = torch.distributions.categorical.Categorical(p)
        return dist.log_prob(actions)
    except Exception as e:
        raise ValueError("Numerical error")
class Base(nn.Module):
    """docstring for Base"""
    def __init__(self, obs_space, num_actions, hidden_size):
        super(Base, self).__init__()

        self.obs_space = obs_space  
        self.num_actions = num_actions

        # for input_view
        self.l1 = nn.Linear(obs_space, hidden_size)

        # for input_act_prob
        # print(num_actions)
        self.l2 = nn.Linear(num_actions, 32)

        # self.l3 = nn.Linear(64, 32)

    def forward(self, obs, input_act_prob):
        # flatten_view = torch.FloatTensor(input_view)
        # print(input_act_prob)
        h_obs = F.relu(self.l1(obs))

        emb_prob = F.relu(self.l2(input_act_prob))
        # dense_prob = F.relu(self.l3(emb_prob))

        concat_layer = torch.cat([h_obs, emb_prob], dim=1)
        # print(concat_layer)
        return concat_layer

class Actor(nn.Module):
    """docstring for Actor"""
    def __init__(self, hidden_size, num_actions):
        super(Actor, self).__init__()
        # print(32 + 2 * hidden_size)
        self.l1 = nn.Linear(32 + hidden_size, hidden_size * 2)
        self.l2 = nn.Linear(hidden_size * 2, num_actions)

    def forward(self, concat_layer):
        dense = F.relu(self.l1(concat_layer))
        logit = self.l2(dense / 0.1)
        policy = F.softmax(logit, dim=-1)
        policy = policy.clamp(1e-10, 1-1e-10)
        return logit, policy
loss_func = nn.CrossEntropyLoss()

class Critic(nn.Module):
    """docstring for Critic"""
    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(32 + hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, concat_layer):
        dense = F.relu(self.l1(concat_layer))
        value = self.l2(dense)
        value = value.reshape(-1)
        return value

class MFAC:
    """docstring for MFAC"""
    def __init__(self, env, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=512, learning_rate=1e-4):
        self.env = env

        self.obs_space = env.n_s
        self.num_actions = env.n_a
        self.reward_decay = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        hidden_size = 128
        self.base = Base(self.obs_space, self.num_actions, hidden_size).to(device)
        self.actor = Actor(hidden_size, self.num_actions).to(device)
        self.critic = Critic(hidden_size).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.base.parameters(),   'lr': learning_rate},
            {'params': self.actor.parameters(),  'lr': learning_rate},
            {'params': self.critic.parameters(), 'lr': learning_rate}
            ])

    @property
    def vars(self):
        return [self.base, self.actor, self.critic]

    def soft(self, **kwargs):
        input_view = torch.FloatTensor(kwargs['state']).to(device)
        input_act_prob = torch.FloatTensor(kwargs['prob']).to(device)
        concat_layer = self.base(input_view, input_act_prob)
        logit, policy = self.actor(concat_layer)
        # print(policy.size()[1])
        return policy  
    def act(self, **kwargs):
        # print(kwargs['state'])
        input_view = torch.FloatTensor(kwargs['state']).to(device)
        input_act_prob = torch.FloatTensor(kwargs['prob']).to(device)
        concat_layer = self.base(input_view, input_act_prob)
        logit, policy = self.actor(concat_layer)
        action = torch.multinomial(policy, 1) # 根据权重选择一次
        action = action.cpu().numpy()
        return action.astype(np.int32).reshape((-1,))

    def train(self, eps=0, eps_attack=0):
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        view = torch.FloatTensor(n,self.obs_space).to(device) # , *
        action = torch.LongTensor(n).to(device)
        reward = torch.FloatTensor(n).to(device)
        act_prob_buff = torch.FloatTensor(n, self.num_actions).to(device)

        ct = 0
        gamma = self.reward_decay
        # collect episodes from multiple separate buffers to a continuous buffer
        for k, episode in enumerate(batch_data):
            v, a, r, prob = episode.views,  episode.actions, episode.rewards, episode.probs
            v = torch.FloatTensor(v).to(device)
            r = torch.FloatTensor(r).to(device)
            a = torch.LongTensor(a).to(device)
            prob = torch.FloatTensor(prob).to(device)

            m = len(episode.rewards)
            assert len(episode.probs) > 0

            concat_layer = self.base(v[-1].reshape(1, -1), prob[-1].reshape(1, -1))
            keep = self.critic(concat_layer)[0] 
            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            act_prob_buff[ct:ct + m] = prob
            ct += m
        assert n == ct

        # train
        concat_layer = self.base(view, act_prob_buff)
        # print(concat_layer.shape)
        value = self.critic(concat_layer)
        logit, policy = self.actor(concat_layer)
        # prob_ = torch.clamp(F.softmax(logit, dim=1), 1e-6, 1)
        # print(prob.shape)
        labels = torch.multinomial(policy, 1)#policy.multinomial(1).data
        action_mask = F.one_hot(action, self.num_actions)
        advantage = (reward - value).detach()
        log_policy = torch.log(policy + 1e-6)
        log_prob = torch.sum(log_policy * action_mask, dim=1) 
        pg_loss = -torch.mean(advantage * log_prob) 
        vf_loss = self.value_coef * torch.mean(torch.square(reward.detach() - value)) 
        neg_entropy = self.ent_coef * torch.mean(torch.sum(policy * log_policy, dim=1))
        total_loss =pg_loss + vf_loss + neg_entropy # + loss_adv
        loss_adv = total_loss
        
        attack_config={"params": {"niters": 10,"epsilon":0.1},'method':'fgsm'}
        if eps_attack > 0 :
            view_cpu = view.cpu().data.numpy()
            act_prob_buff_cpu = act_prob_buff.cpu().data.numpy()
            adv_state = torch.tensor(attack(self, view_cpu, act_prob_buff_cpu,eps, attack_config)).to(device)
            
            # print(self.actor(self.base(adv_state, act_prob_buff)).size())
            # print(labels)
            loss_adv = eps_attack*torch.mean(loss_func(self.actor(self.base(adv_state, act_prob_buff))[1], labels.reshape((-1,)))) # 
            total_loss = total_loss + loss_adv
        # train op (clip gradient)
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.base.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        print('[*] PG_LOSS:', np.round(pg_loss.item(), 6), '/ VF_LOSS:', np.round(vf_loss.item(), 6), 
        '/ ENT_LOSS:', np.round(neg_entropy.item(), 6), '/ ADV_LOSS:', np.round(loss_adv.item(), 6), 
        '/ VALUE:', np.mean(value.cpu().detach().numpy()),'/ total_loss:', total_loss)


    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def save(self, dir_path, id = 0):

        model_vars = {
            'base':   self.base.state_dict(),
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

        file_path = os.path.join(dir_path, "mfac-"+str(id)+".pth")
        torch.save(model_vars, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfac-"+str(step)+".pth")
        print(dir_path)
        print(file_path)
        model_vars = torch.load(file_path)

        self.base.load_state_dict(model_vars['base'])
        self.actor.load_state_dict(model_vars['actor'])
        self.critic.load_state_dict(model_vars['critic'])

        print("[*] Loaded model from {}".format(file_path))
