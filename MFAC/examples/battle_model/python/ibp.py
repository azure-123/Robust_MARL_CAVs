import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def base_network_bounds(model, view, feature, act_prob_buff, epsilon):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    # view = view.reshape(-1, np.prod(view.shape))
    flatten_view = view.reshape(-1, np.prod([13,13,7]))
    view_upper = flatten_view + epsilon
    view_lower = flatten_view - epsilon
    feature_upper = feature + epsilon
    feature_lower = feature - epsilon

    layer = [i for i in model.modules()]
    # model(view, feature, act_prob_buff)
    # model.l1(view_upper)
    upper_2 = lower_2 = F.relu(layer[4](F.relu(layer[3](act_prob_buff))))
    # upper_2, lower_2 = base_weighted_bound(layer[3], act_prob_buff, act_prob_buff)
    # upper_2, lower_2 = F.relu(upper_2), F.relu(lower_2)
    # upper_2, lower_2 = base_weighted_bound(layer[4], upper_2, lower_2)
    # upper_2, lower_2 = F.relu(upper_2), F.relu(lower_2)
    upper_1, lower_1 = weighted_bound(layer[2], feature_upper, feature_lower)
    upper_1, lower_1 = F.relu(upper_1), F.relu(lower_1)
    upper_0, lower_0 = weighted_bound(layer[1], view_upper, view_lower)
    upper_0, lower_0 = F.relu(upper_0), F.relu(lower_0)
    # print(lower_2)
    # print(upper_0.shape)
    upper = torch.cat([upper_0, upper_1, upper_2], dim=1)
    lower = torch.cat([lower_0, lower_1, lower_2], dim=1)

    return upper, lower



def network_bounds(model, upper, lower):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    layer = [layer for layer in model.modules()]
    # print(layer)
    upper, lower = weighted_bound(layer[1], upper, lower)
    upper, lower = F.relu(upper), F.relu(lower)
    upper, lower = weighted_bound(layer[2], upper/0.1, lower/0.1)
    # upper, lower = F.softmax(upper), F.softmax(lower)
    
    return upper, lower
