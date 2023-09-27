from cmath import inf
import torch
import torch.nn.functional as F
from scipy.special import lambertw
import numpy as np
import math

import time
import sys
sys.path.append('/nethome/ericwong/attack/wasserstein_attack')
from lambertw import lambertw 
from scipy.special import lambertw as lambertw_np
# lambertw is not implemented in pytorch 

TYPE_2NORM = '2norm'
TYPE_CONJUGATE = 'conjugate'
TYPE_PLAN = 'plan'

MAX_FLOAT = 1e38# 1.7976931348623157e+308

def any_nan(X): 
    return (X != X).any().item()
def any_inf(X): 
    return (X == float('inf')).any().item()

def lamw_np(x): 
    cuda = x.is_cuda
    x = x.cpu().numpy()
    I = x > 1e-10
    y = np.copy(x)
    y[I] = np.real(lambertw_np(x[I])) # 返回复数参数的实部
    out = torch.from_numpy(np.real(y))
    if cuda: 
        out = out.cuda()
    return out

def lamw(x): 
    I = x > 1e-10
    y = torch.clone(x)
    y[I] = lambertw(x[I])
    return y

# batch dot product
def _bdot(X,Y): 
    return torch.matmul(X.unsqueeze(-2), Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def _unfold(x, kernel_size, padding=None): 
    # this is necessary because unfold isn't implemented for multidimensional batches
    size = x.size()
    if len(size) > 4: 
        x = x.contiguous().view(-1, *size[-3:])
    out = F.unfold(x, kernel_size, padding=kernel_size//2)
    if len(size) > 4: 
        out = out.view(*size[:-3], *out.size()[1:])
    return out


def _mm(A,x, shape): 
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = _unfold(x, kernel_size, padding=kernel_size//2).transpose(-1,-2)
    unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3)
    out = torch.matmul(unfolded, collapse2(A.contiguous()).unsqueeze(-1)).squeeze(-1)

    return unflatten2(out)

def _mm_1(A,x, shape): 
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = x
    # unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1)))
    out = torch.matmul(unfolded, collapse0(A.contiguous()).unsqueeze(-1))
    return unflatten1(out)

def wasserstein_cost(X, p=2, kernel_size=5):
    if kernel_size % 2 != 1: 
        raise ValueError("Need odd kernel size")
        
    center = kernel_size // 2
    C = X.new_zeros(kernel_size,kernel_size)
    for i in range(kernel_size): 
        for j in range(kernel_size): 
            C[i,j] = (abs(i-center)**2 + abs(j-center)**2)**(p/2)
    return C

def unsqueeze3(X):
    return X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def unsqueeze1(X):
    return X.unsqueeze(-1)

def collapse0(X): 
    return X.view(*X.size()[:0], -1)

def collapse1(X): 
    return X.view(*X.size()[:-1], -1)

def collapse2(X): 
    return X.view(*X.size()[:-2], -1)

def collapse3(X): 
    return X.reshape(*X.size()[:-3], -1)

def _expand(X, shape): 
    return X.view(*X.size()[:-1], *shape)

def unflatten1(X): 
    # print('unflatten2', X.size()) #[8, 13, 91]
    n = X.size(-1)
    # k = int(math.sqrt(n))
    return _expand(X,n)

def unflatten2(X): 
    # print('unflatten2', X.size()) #[8, 13, 91]
    n = X.size(-1)
    k = int(math.sqrt(n))
    return _expand(X,(k,k))

def _expand_filter(X, nfilters): 
    sizes = list(-1 for _ in range(X.dim()))
    sizes[-3] = nfilters
    return X.expand(*sizes)


def bdot(x,y): 
    return _bdot(collapse3(x), collapse3(y))

def projected_sinkhorn(*args, **kwargs): 
    return log_sinkhorn(*args, objective='2norm', **kwargs)

def projected_sinkhorn_lw(*args, **kwargs): 
    return log_sinkhorn_lw(*args, objective='2norm', **kwargs)


def conjugate_sinkhorn(*args, **kwargs): 
    return log_sinkhorn(*args, objective='conjugate', **kwargs)

def log_sinkhorn(X, Y, C, epsilon, lam, verbose=False, plan=False,
   objective='2norm', maxiters=50, return_objective=False):  
    """ 
    if objective == '2norm': 
        minimize_Z ||Y-Z||_2 subject to Z in Wasserstein ball around X 

        we return Z

    if objective == 'conjugate': 
        minimize_Z -Y^TZ subject to Z in Wasserstein ball around X

        however instead of Z we return the dual variables u,psi for the 
        equivalent objective: 
        minimize_{u,psi} 

    Inputs: 
        X : batch size x nfilters x image width x image height
        Y : batch size x noutputs x total input dimension

        these really out to be broadcastable

        Same as above but in log space. Note that the implementation follows the 
        rescaled version -\lambda Y^TZ + entropy(W) instead of 
        -Y^TZ + 1/lambda * entropy(W), so the final objective is downscaled by lambda
    """
    batch_sizes = X.size()[:-3]
    #print(X.size())
    nfilters = X.size(-3)

    # size check
    for xd,yd in (zip(reversed(X.size()),reversed(Y.size()))): 
        assert xd == yd or xd == 1 or yd == 1

    # helper functions
    expand3 = lambda x: _expand(x, X.size()[-3:])
    expand_filter = lambda x: _expand_filter(x, X.size(-3))
    mm = lambda A,x: _mm(expand_filter(A),x,X.size())
    norm = lambda x: torch.norm(collapse3(x), dim=-1)
    # like numpy
    allclose = lambda x,y: (x-y).abs() <= 1e-4 + 1e-4*y.abs()
    
    # assert valid distributions
    # print(X)
    # assert (X>=0).all()
    if ~((X>=0).all()):
        X.data = torch.clamp(X.data, 0, 1)
    # print(X.size())
    # assert ((collapse3(X).sum(-1) - 1).abs() <= 1e-4).all()
    assert X.dim() == Y.dim()


    size = tuple(max(sx,sy) for sx,sy in zip(X.size(), Y.size()))

    # total dimension size for each example
    m = collapse3(X).size(-1)
    
    if objective == TYPE_CONJUGATE: 
        alpha = torch.log(X.new_ones(*size)/m) + 0.5
        exp_alpha = torch.exp(-alpha)
        beta = -lam*Y.expand_as(alpha).contiguous()
        exp_beta = torch.exp(-beta)

        # check for overflow
        if (exp_beta == float('inf')).any(): 
            print(beta.min())
            raise ValueError('Overflow error: in logP_sinkhorn for e^beta')


        # EARLY TERMINATION CRITERIA: if the nu_1 and the 
        # center of the ball have no pixels with overlapping filters, 
        # thenthe wasserstein ball has no effect on the objective. 
        # Consequently, we should just return the objective 
        # on the center of the ball. Notably, if the filters don't overlap, 
        # then the pixels themselves don't either, so we can conclude that 
        # the objective is 0. 

        # We can detect overlapping filters by applying the cost 
        # filter and seeing if the sum is 0 (e.g. X*C*Y)
        C_tmp = C.clone() + 1
        while C_tmp.dim() < Y.dim(): 
            C_tmp = C_tmp.unsqueeze(0)
        I_nonzero = bdot(X,mm(C_tmp,Y)) != 0
        I_nonzero_ = unsqueeze3(I_nonzero).expand_as(alpha)

        def eval_obj(alpha, exp_alpha, psi, K): 
            return -psi*epsilon - bdot(torch.clamp(alpha,max=MAX_FLOAT),X) - bdot(exp_alpha, mm(K, exp_beta))

        def eval_z(alpha, exp_alpha, psi, K): 
            return exp_beta*mm(K, exp_alpha)

        psi = X.new_ones(*size[:-3])
        K = torch.exp(-unsqueeze3(psi)*C - 1)

        old_obj = -float('inf')
        i = 0

        with torch.no_grad(): 
            while True: 
                alpha[I_nonzero_] = (torch.log(mm(K,exp_beta)) - torch.log(X))[I_nonzero_]
                exp_alpha = torch.exp(-alpha)

                dpsi = -epsilon + bdot(exp_alpha,mm(C*K,exp_beta))
                ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
                delta = dpsi/ddpsi

                psi0 = psi
                t = X.new_ones(*delta.size())
                neg = (psi - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi - t*delta < 0
                psi[I_nonzero] = torch.clamp(psi - t*delta, min=0)[I_nonzero]

                K = torch.exp(-unsqueeze3(psi)*C - 1)

                # check for convergence
                obj = eval_obj(alpha, exp_alpha, psi, K)
                if verbose: 
                    print('obj', obj)
                i += 1
                if i > maxiters or allclose(old_obj,obj).all(): 
                    if verbose: 
                        print('terminate at iteration {}'.format(i))
                    break

                old_obj = obj

        if return_objective: 
            obj = -bdot(X,Y)
            obj[I_nonzero] = eval_obj(alpha, exp_alpha, psi, K)[I_nonzero]
            return obj
        else: 
            z = eval_z(alpha, exp_alpha, psi,K)
            z[~I_nonzero] = 0
            return z

    elif objective == TYPE_2NORM: 
        alpha = torch.log(X.new_ones(*size)/m)
        beta = torch.log(X.new_ones(*size)/m)
        # print(beta)
        exp_alpha = torch.exp(-alpha)
        exp_beta = torch.exp(-beta)

        psi = X.new_ones(*size[:-3])
        K = torch.exp(-unsqueeze3(psi)*C - 1)

        def eval_obj(alpha, beta, exp_alpha, exp_beta, psi, K): 
            return (-0.5/lam*bdot(beta,beta) - psi*epsilon 
                    - bdot(torch.clamp(alpha,max=1e10),X) 
                    - bdot(torch.clamp(beta,max=1e10),Y)
                    - bdot(exp_alpha, mm(K, exp_beta)))

        old_obj = -float('inf')
        i = 0

        if verbose:
            start_time = time.time()

        with torch.no_grad(): 
            while True: 
                alphat = norm(alpha)
                betat = norm(beta)

                alpha = (torch.log(mm(K,exp_beta)) - torch.log(X))
                exp_alpha = torch.exp(-alpha)
                
                beta = lamw(lam*torch.exp(lam*Y)*mm(K,exp_alpha)) - lam*Y
                exp_beta = torch.exp(-beta)

                dpsi = -epsilon + bdot(exp_alpha,mm(C*K,exp_beta))
                ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
                delta = dpsi/ddpsi

                psi0 = psi
                t = X.new_ones(*delta.size())
                neg = (psi - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi - t*delta < 0
                psi = torch.clamp(psi - t*delta, min=0)

                # update K
                K = torch.exp(-unsqueeze3(psi)*C - 1)

                # check for convergence
                obj = eval_obj(alpha, exp_alpha, beta, exp_beta, psi, K) 
                i += 1
                if i > maxiters or allclose(old_obj,obj).all(): 
                    if verbose: 
                        print('terminate at iteration {}'.format(i), maxiters)
                        if i > maxiters: 
                            print('warning: took more than {} iters'.format(maxiters))
                    break
                old_obj = obj
        return beta/lam + Y

def log_sinkhorn_lw(X, Y, C, epsilon_1,epsilon_2 , lam, verbose=True, plan=False,
   objective='2norm', maxiters=50, return_objective=False):  
    """ 
    if objective == '2norm': 
        minimize_Z ||Y-Z||_2 subject to Z in Wasserstein ball around X 

        we return Z

    if objective == 'conjugate': 
        minimize_Z -Y^TZ subject to Z in Wasserstein ball around X

        however instead of Z we return the dual variables u,psi for the 
        equivalent objective: 
        minimize_{u,psi} 

    Inputs: 
        X : batch size x nfilters x image width x image height
        Y : batch size x noutputs x total input dimension

        these really out to be broadcastable

        Same as above but in log space. Note that the implementation follows the 
        rescaled version -\lambda Y^TZ + entropy(W) instead of 
        -Y^TZ + 1/lambda * entropy(W), so the final objective is downscaled by lambda
    """
    batch_sizes = X.size()[:-3]
    #print(X.size())
    nfilters = X.size(-3)

    # size check
    for xd,yd in (zip(reversed(X.size()),reversed(Y.size()))): 
        assert xd == yd or xd == 1 or yd == 1

    # helper functions
    expand3 = lambda x: _expand(x, X.size()[-3:])
    expand_filter = lambda x: _expand_filter(x, X.size(-3))
    mm = lambda A,x: _mm(expand_filter(A),x,X.size())
    norm = lambda x: torch.norm(collapse3(x), dim=-1)
    # like numpy
    allclose = lambda x,y: (x-y).abs() <= 1e-4 + 1e-4*y.abs()
    
    # assert valid distributions
    # print(X)
    # assert (X>=0).all()
    if ~((X>=0).all()):
        X.data = torch.clamp(X.data, 0, 1)
    # print(X.size())
    # assert ((collapse3(X).sum(-1) - 1).abs() <= 1e-4).all()
    assert X.dim() == Y.dim()


    size = tuple(max(sx,sy) for sx,sy in zip(X.size(), Y.size()))

    # total dimension size for each example
    m = collapse3(X).size(-1)
    
    if objective == TYPE_CONJUGATE: 
        a = 1
        # alpha = torch.log(X.new_ones(*size)/m) + 0.5
        # exp_alpha = torch.exp(-alpha)
        # beta = -lam*Y.expand_as(alpha).contiguous()
        # exp_beta = torch.exp(-beta)

        # # check for overflow
        # if (exp_beta == float('inf')).any(): 
        #     print(beta.min())
        #     raise ValueError('Overflow error: in logP_sinkhorn for e^beta')


        # # EARLY TERMINATION CRITERIA: if the nu_1 and the 
        # # center of the ball have no pixels with overlapping filters, 
        # # thenthe wasserstein ball has no effect on the objective. 
        # # Consequently, we should just return the objective 
        # # on the center of the ball. Notably, if the filters don't overlap, 
        # # then the pixels themselves don't either, so we can conclude that 
        # # the objective is 0. 

        # # We can detect overlapping filters by applying the cost 
        # # filter and seeing if the sum is 0 (e.g. X*C*Y)
        # C_tmp = C.clone() + 1
        # while C_tmp.dim() < Y.dim(): 
        #     C_tmp = C_tmp.unsqueeze(0)
        # I_nonzero = bdot(X,mm(C_tmp,Y)) != 0
        # I_nonzero_ = unsqueeze3(I_nonzero).expand_as(alpha)

        # def eval_obj(alpha, exp_alpha, psi, K): 
        #     return -psi*epsilon - bdot(torch.clamp(alpha,max=MAX_FLOAT),X) - bdot(exp_alpha, mm(K, exp_beta))

        # def eval_z(alpha, exp_alpha, psi, K): 
        #     return exp_beta*mm(K, exp_alpha)

        # psi = X.new_ones(*size[:-3])
        # K = torch.exp(-unsqueeze3(psi)*C - 1)

        # old_obj = -float('inf')
        # i = 0

        # with torch.no_grad(): 
        #     while True: 
        #         alpha[I_nonzero_] = (torch.log(mm(K,exp_beta)) - torch.log(X))[I_nonzero_]
        #         exp_alpha = torch.exp(-alpha)

        #         dpsi = -epsilon + bdot(exp_alpha,mm(C*K,exp_beta))
        #         ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
        #         delta = dpsi/ddpsi

        #         psi0 = psi
        #         t = X.new_ones(*delta.size())
        #         neg = (psi - t*delta < 0)
        #         while neg.any() and t.min().item() > 1e-2:
        #             t[neg] /= 2
        #             neg = psi - t*delta < 0
        #         psi[I_nonzero] = torch.clamp(psi - t*delta, min=0)[I_nonzero]

        #         K = torch.exp(-unsqueeze3(psi)*C - 1)

        #         # check for convergence
        #         obj = eval_obj(alpha, exp_alpha, psi, K)
        #         if verbose: 
        #             print('obj', obj)
        #         i += 1
        #         if i > maxiters or allclose(old_obj,obj).all(): 
        #             if verbose: 
        #                 print('terminate at iteration {}'.format(i))
        #             break

        #         old_obj = obj

        # if return_objective: 
        #     obj = -bdot(X,Y)
        #     obj[I_nonzero] = eval_obj(alpha, exp_alpha, psi, K)[I_nonzero]
        #     return obj
        # else: 
        #     z = eval_z(alpha, exp_alpha, psi,K)
        #     z[~I_nonzero] = 0
        #     return z

    elif objective == TYPE_2NORM: 
        alpha = torch.log(X.new_ones(*size)/m)
        beta = torch.log(X.new_ones(*size)/m)
        # print(beta)
        exp_alpha = torch.exp(-alpha)
        exp_beta = torch.exp(-beta)
        # print(exp_alpha.shape)
        # print(exp_beta.shape)

        psi_1 = X.new_ones(*size[:-3])
        psi_2 = X.new_ones(*size[:-3])
        # print(psi_1.shape)
        def A_(z, x):
            AA = X.new_ones(*size)
            # print(AA.shape)
            # print(x.shape[0])
            # print((z[0]-x[0]).shape)
            for i in range (x.shape[0]):
                max_inf = torch.norm(z[0]-x[0], p = np.inf)
                for j in range (x.shape[1]):
                    for k in range(x.shape[2]):
                        for l in range(x.shape[3]):
                            # print(AA[i][j][k][l])
                            if torch.abs(z[i][j][k][l]-x[i][j][k][l]) == max_inf:
                                if z[i][j][k][l]-x[i][j][k][l] >=0:
                                    AA[i][j][k][l] = 1
                                else:
                                    AA[i][j][k][l] = -1
                            else:
                                AA[i][j][k][l] = 0
            return AA  
        # 初始化A 初始化z
        A  = X.new_ones(*size) 
        z = (beta - A.mul(unsqueeze3(psi_2)))/lam + Y  
        # z = Y
        # A = A_(z, X)

        K = torch.exp(-unsqueeze3(psi_1)*C - 1)

        def eval_obj(alpha, beta, exp_alpha, exp_beta, psi_1, psi_2, K, A): 

            inf_ = torch.ones_like(psi_2)

            for i in range (psi_2.shape[0]):
                inf_[i] = torch.norm(Y[i] - X[i] + (beta[i]-A[i].mul(psi_2[i].unsqueeze(-1).unsqueeze(-1)))/lam ,p = np.inf)
            # print(inf_)
            return (-0.5/lam*bdot(beta-A.mul(unsqueeze3(psi_2)),beta-A.mul(unsqueeze3(psi_2))) - psi_1*epsilon_1 - psi_2*epsilon_2
                    - bdot(torch.clamp(alpha,max=1e10),X) 
                    - bdot(torch.clamp(beta,max=1e10),Y)
                    - bdot(exp_alpha, mm(K, exp_beta))
                    + psi_2.mul(inf_))

        old_obj = -float('inf')
        ii = 0

        if verbose:
            start_time = time.time()

        with torch.no_grad(): 
            while True: 
                alphat = norm(alpha)
                betat = norm(beta)
                # A = A_(z, X)
                # 更新 alpha
                alpha = (torch.log(mm(K,exp_beta)) - torch.log(X))
                exp_alpha = torch.exp(-alpha)
                # 更新 beta
                beta = lamw(lam*torch.exp(lam*Y-3*A.mul(unsqueeze3(psi_2)))*mm(K,exp_alpha)) - lam*Y + 3*A.mul(unsqueeze3(psi_2))
                exp_beta = torch.exp(-beta)

                # 更新psi_1 牛顿法
                dpsi = -epsilon_1 + bdot(exp_alpha,mm(C*K,exp_beta))
                ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
                delta = dpsi/ddpsi
                # print(dpsi.shape)
                # print(ddpsi.shape)
                psi0 = psi_1
                t = X.new_ones(*delta.size())
                # print(delta.shape)
                # print(t.shape)
                neg = (psi_1 - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi_1 - t*delta < 0
                psi_1 = torch.clamp(psi_1 - t*delta, min=0)

                # 更新psi_2 不能用牛顿法 梯度下降就好
                # psi_2 = psi_2_(psi_2)
                # psi_2 = torch.ones_like(dpsi)
                # psi_2 = torch.where(psi_2 == 1, torch.zeros_like(psi_2)+1000, torch.zeros_like(psi_2)+1000)
                dpsi_2 = torch.ones_like(dpsi)
                for i in range (dpsi_2.shape[0]):
                    dpsi_2[i] = torch.norm(z[i] - X[i], p = np.inf) -epsilon_2 
                # print(dpsi_2)
                dpsi_2 = unsqueeze3(dpsi_2) #- A.mul(unsqueeze3(dpsi_2))

                # ddpsi_2 = torch.where(A == 0, torch.zeros_like(A)+5e-3, A)
                delta_2 = dpsi_2 #/ddpsi_2 
                # print(delta_2.shape)
                t_2 = X.new_ones(*delta_2.size())
                # print(t_2.shape)
                # print(psi_1.shape,psi_2.shape)
                # print(psi_2)
                neg_2 = (psi_2 - bdot(t_2, delta_2)< 0)
                # print(neg_2)
                while neg_2.any() and t_2.min().item() > 1e-2:
                    t_2[neg_2] /= 2
                    neg_2 = (psi_2 - bdot(t_2, delta_2)) < 0
                psi_2 = torch.clamp(psi_2 - bdot(t_2, delta_2), min = 0)
                # print(psi_2)
                # # update K
                K = torch.exp(-unsqueeze3(psi_1)*C - 1)
                # 更新 A
                z_old = z
                z = (beta - A.mul(unsqueeze3(psi_2)))/lam + Y
                # print(beta.max(),beta.min())
                # print(A.mul(unsqueeze3(psi_2)).min(), A.mul(unsqueeze3(psi_2)).max())
                # print((z - z_old).min(),(z - z_old).max())
                A_old = A
                A = A_(z, X)
                # print((A - A_old).min(),(A - A_old).max())
                # check for convergence
                obj = eval_obj(alpha, exp_alpha, beta, exp_beta, psi_1, psi_2, K, A) 
                # print(obj.shape)
                # print(obj)
                ii += 1
                if ii > maxiters or allclose(old_obj,obj).all(): 
                    # print(psi_2)
                    if verbose: 
                        print('terminate at iteration {}'.format(ii), maxiters)
                        if ii > maxiters: 
                            print('warning: took more than {} iters'.format(maxiters))
                    break
                old_obj = obj
        return z#(beta - A.mul(unsqueeze3(psi_2)))/lam + Y
