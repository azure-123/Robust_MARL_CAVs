import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

def tar_attack(model, epsilon, states, action, tar_action, opt, use_cuda):
    loss_func = nn.CrossEntropyLoss()
    adv_states = torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32)
    if use_cuda:
        adv_states = adv_states.cuda()
    logits = model(adv_states)

    if use_cuda:
        loss = -loss_func(logits[0], torch.tensor(np.array(tar_action)).long().cuda()) + loss_func(logits[0], torch.tensor(np.array(action)).long().cuda())
    else:
        loss = -loss_func(logits[0], torch.tensor(np.array(tar_action)).long()) + loss_func(logits[0], torch.tensor(np.array(action)).long())
    
    adv_states.retain_grad()
    opt.zero_grad()
    loss.backward()
    

    eta_0 = epsilon * adv_states.grad.data.sign()
    adv_states.data = Variable(adv_states.data + eta_0, requires_grad=True)

    if use_cuda:
        eta_0 = torch.clamp(adv_states.data - torch.tensor(np.array(states)).cuda().data, -epsilon, epsilon)
        adv_states.data = torch.tensor(np.array(states)).cuda().data + eta_0
    else:
        eta_0 = torch.clamp(adv_states.data - torch.tensor(np.array(states)).data, -epsilon, epsilon)
        adv_states.data = torch.tensor(np.array(states)).data + eta_0

    return adv_states[0].cpu().data.numpy()