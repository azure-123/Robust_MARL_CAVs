import numpy as np
import os
import datetime


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'


class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError


class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value


class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.views = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.terminal = False

    def append(self, view,  action, reward,  probs=None, terminal=False):
        self.views.append(view.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs)
        self.terminal = terminal



class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        terminal = kwargs['terminal']

        if self.use_mean:
            probs = kwargs['prob']

        buffer = self.buffer
        index = np.random.permutation(len(view))

        for i in range(len(view)):
            i = index[i]
            entry = buffer.get(i)
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[i] = entry

            if self.use_mean:
                entry.append(view[i], acts[i], rewards,  probs=probs[i], terminal = terminal)
            else:
                entry.append(view[i],  acts[i], rewards, terminal = terminal)

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()


class AgentMemory(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, use_mean=False):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean

        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)

    def append(self, obs0, feat0, act, reward, alive, prob=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=np.bool))

        if self.use_mean:
            self.prob.append(np.array([prob]))

    def pull(self):
        res = {
            'obs0': self.obs0.pull(),
            'feat0': self.feat0.pull(),
            'act': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob': None if not self.use_mean else self.prob.pull()
        }

        return res


class MemoryGroup(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, use_mean=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_n = act_n

        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.obs0.append(kwargs['obs0'])
        self.feat0.append(kwargs['feat0'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])

        if self.use_mean:
            self.prob.append(kwargs['prob'])

        mask = np.where(kwargs['terminals'] == True, False, True)
        mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len, use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i], prob=kwargs['prob'][i])
            else:
                self.agent[_id].append(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i])

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            self._new_add += len(tmp['obs0'])
            self._flush(**tmp)
        self.agent = dict()  # clear

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries

        obs = self.obs0.sample(idx)
        obs_next = self.obs0.sample(next_idx)
        feature = self.feat0.sample(idx)
        feature_next = self.feat0.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
            return obs, feature, actions, act_prob, obs_next, feature_next, act_next_prob, rewards, dones, masks
        else:
            return obs, feature, obs_next, feature_next, dones, rewards, actions, masks

    def get_batch_num(self):
        print('\n[INFO] Length of buffer and new add:', len(self.obs0), self._new_add)
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.obs0)


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name):
        import tensorboard_logger
        from tensorboard_logger import configure, log_value

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        tensorboard_logger.clean_default_logger()
        unique_token = "{}__{}".format(log_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        configure(log_dir + "/" + unique_token)
        self.train_writer = log_value

    def write(self, summary_dict, step):
        """
        summary_dict: dict, summary value dict
        step: int, global step
        """
        assert isinstance(summary_dict, dict)
        for key, value in summary_dict.items():
            self.train_writer(key, value, step)

# (env, args.max_steps, models, play, render=args.render, save_every=args.save_every, tau=0.01, 
# log_name=args.algo,
# log_dir=log_dir, model_dir=model_dir, train=True)
class Runner(object):
    def __init__(self, env, max_steps, models,
                play_handle, render=False, save_every=None, tau=None, log_name=None, log_dir=None, model_dir=None, train=False):
        """Initialize runner

        Parameters
        ----------
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.render = render
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train
        self.tau = tau

        if self.train:
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

    def run(self, variant_eps, iteration, eps_attack, total_reward=None , avg_speed=None, crash_ra=None):
        info = None
        # pass
        info = {'total_reward': 0., 'avg_speed': 0. , 'crash': 0.}
        # print(eps_attack)
        total_rewards, speed, crash = self.play(env=self.env, n_round=iteration,
                                                max_steps=self.max_steps, models=self.models, 
                                                print_every=25, eps=variant_eps, render=self.render, train=self.train,eps_attack=eps_attack)
        # print(crash)
        
        info['total_reward'] = total_rewards
        info['avg_speed'] = speed
        info['crash'] = True if crash>0 else False

        if self.train:
            print('\n[INFO] {}'.format(info))
            self.summary.write(info, iteration)

            if self.save_every and (iteration + 1) % self.save_every == 0:
                print(Color.INFO.format('[INFO] Saving model ...'))
                self.models.save(self.model_dir, iteration + 1)#,step=iteration
                
        else:
            print('\n[INFO] {}'.format(info))
            total_reward.append(info['total_reward'])
            avg_speed.append(info['avg_speed'])
            crash_ra.append(info['crash'])

            
            


