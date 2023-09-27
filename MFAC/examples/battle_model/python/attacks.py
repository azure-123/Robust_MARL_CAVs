from locale import normalize
# from math import dist
from random import random
import torch
from torch import autograd
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
# from layers import SinkhornDistance

# from utils import arctanh_rescale, tanh_rescale, to_one_hot, calc_kl
# from projected_sinkhorn import conjugate_sinkhorn, projected_sinkhorn, wasserstein_cost, projected_sinkhorn_lw#, projected_sinkhorn_1

TARGET_MULT = 10000.0

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
Variable = lambda *args, **kwargs:  autograd.Variable(*args, **kwargs)

# CARTPOLE_STD=[0.7322321, 1.0629482, 0.12236707, 0.43851405]
# ACROBOT_STD=[0.36641926, 0.65119815, 0.6835106, 0.67652863, 2.0165246, 3.0202584]


def fgsm(model, X, former_act_prob,eps, y, verbose=False, params={}):
    epsilon=params.get('epsilon', 0.05)
    X = torch.tensor(X,dtype=torch.float)#.to(device)
    X_adv = Variable(X.data, requires_grad=True)#
    logits = model.soft(state=X_adv, prob=former_act_prob, eps=eps)
    # loss = F.nll_loss(logits, y)
    # print(logits, torch.LongTensor(y))
    loss = F.cross_entropy(logits,torch.LongTensor(y).to(device))
    # print(loss)
    model.optimizer.zero_grad()
    loss.backward()
    eta = epsilon*X_adv.grad.data.sign()
    X_adv_0 = Variable(X_adv.data + eta, requires_grad=True)
    eta_0 = torch.clamp(X_adv_0.data- X.data, -epsilon, epsilon)
    X_adv.data = X.data + eta_0

    #print(X_adv_0.data)
    return X_adv.cpu().data.numpy()


def random_nosie(model, X, former_act_prob,eps, y, verbose=False, params={}):
    epsilon=params.get('epsilon', 0.05)
    X = torch.tensor(X,dtype=torch.float)
    X_adv = Variable(X.data, requires_grad=True)
    eta_0 = 2 * epsilon * torch.rand(X.size()) - epsilon

    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- X.data, -epsilon, epsilon)
    X_adv.data = X + eta_0

    #print(X_adv_0.data)
    return X_adv.cpu().data.numpy()



def attack(model, X,former_act_prob,eps, attack_config, loss_func=nn.CrossEntropyLoss()):
    method = attack_config.get('method', 'pgd')
    verbose = attack_config.get('verbose', False)
    params = attack_config.get('params', {})
    params['loss_func'] = loss_func
    y = model.act(state=X, prob=former_act_prob, eps=eps)
    if method == 'fgsm':
        atk = fgsm

    else:
        # y = model.soft(state=X, prob=former_act_prob, eps=eps)
        # y = torch.eye(21)[y,:]
        atk = random_nosie
    adv_X = atk(model, X, former_act_prob,eps,y, verbose=verbose, params=params)
    # print(111111111111)
    # abs_diff = abs(adv_X[0]-X[0])
    # abs_diff1 = abs(adv_X[1]-X[1])
    # print(abs_diff)
    # print(abs_diff1)
    #print('adv image range: {}-{}, ori action: {}, adv action: {}, l1 norm: {}, l2 norm: {}, linf norm: {}'.format(torch.min(adv_X).cpu().numpy(), torch.max(adv_X).cpu().numpy(), model.act(X)[0], model.act(adv_X)[0], np.sum(abs_diff), np.linalg.norm(abs_diff), np.max(abs_diff)))
    return adv_X

