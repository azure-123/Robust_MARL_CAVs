import torch
from torch import nn
from torch.autograd import Variable

def tar_attack(model, epsilon, states, actions, tar_actions):
    loss_func = nn.CrossEntropyLoss()
    

    adv_states = Variable(states.data, requires_grad=True)
    



    # X_adv = Variable(agent_inputs.data, requires_grad=True)

    # logits, hid = model.soft(X_adv, avail_actions, batch, hidden_state, t_ep=t, t_env=t_env, test_mode=True)
    

    # loss = -loss_func(logits[0],tar_actions[0]) + loss_func(logits[0], actions[0])
    # opt.zero_grad()
    # loss.backward(retain_graph=True)

    # eta_0 = epsilon * X_adv.grad.data.sign()
    # X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    # eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    # X_adv.data = agent_inputs.data + eta_0
    # # print(X_adv - X)

    # return X_adv.cpu().data.numpy()