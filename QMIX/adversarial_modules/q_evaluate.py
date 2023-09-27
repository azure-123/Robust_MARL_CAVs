import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QCritic(nn.Module):
    def __init__(self, args):
        super(QCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        input_shape = self.state_dim + self.n_agents 

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64) # 4s 少几层
        self.fc5 = nn.Linear(64, 1)

    def forward(self, actions , state):
        # inputs = self._build_inputs(batch, t=t)
        bs = state.size(0)
        # print(bs)
        # print(state.size())
        # print(actions.size())
        # print(th.cat([state, actions], dim=-1))
        x = F.relu(self.fc1(th.cat([state, actions], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        qx = self.fc5(x)
        q = qx.view(bs, -1, 1)
        return q



