# WARNING/NOTE: use this implemention WITHOUT inplace module following!
# like: `Sequential(BCN(),ReLU(True))`
# or
# `out = BCN()...(x); out += x`
# then input grad will goes WRONG, and weight/bias grad is None.
# I have NO idea to do with this so far.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.nn.init as init
from xcn_cuda import instance_forward, instance_backward, batch_forward, batch_backward


class instance_cond_norm(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, clip, eps):
        out, mean, invstd = instance_forward(input, weight, bias, clip.item(), eps)
        ctx.save_for_backward(input, weight, mean, invstd, clip)

        return out
    
    @staticmethod
    def backward(ctx, grad):
        inp, weight, mean, invstd, clip = ctx.saved_variables
        d_input, d_weight, d_bias = instance_backward(grad, inp, weight, mean, invstd, clip.item())
        return d_input, d_weight, d_bias, None, None



class instCondNorm(nn.Module):
    r'''
    
    WARNING/NOTE: use this implemention WITHOUT inplace module following!
    like: `Sequential(BCN(),ReLU(True))`
    or
    `out = BCN()...(x); out += x`
    then input grad will goes WRONG, and weight/bias grad is None.
    I have NO idea to do with this so far.
    '''
    def __init__(self, planes=64, minl=1.0, eps=1e-5):
        super().__init__()
        self.planes = planes
        self.minl = minl
        self.clip = torch.tensor(minl, dtype=torch.double)
        self.eps = eps
        self.weight = Parameter(torch.ones(planes))
        self.bias = Parameter(torch.zeros(planes))
    
    def forward(self, x):
        return instance_cond_norm.apply(x, self.weight, self.bias, self.clip, self.eps)

    def extra_repr(self):
        s = 'planes={planes}, minl={minl}, epsilon={eps}'
        return s.format(**self.__dict__)
        

class batch_cond_norm(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, clip, eps):
        # x = input - torch.mean(input, dim=[2,3], keepdim=True)
        # norm = torch.sqrt(torch.mean(x**2, dim=[2,3], keepdim=True) + eps)
        # inv_cnorm = torch.clamp_min(norm, clip).reciprocal()
        # x_norm = x * inv_cnorm
        # out = weight * x_norm + bias
        out, mean, invstd = batch_forward(input, weight, bias, clip.item(), eps)
        ctx.save_for_backward(input, weight, mean, invstd, clip)

        return out
    
    @staticmethod
    def backward(ctx, grad):
        # x, clip, inv_cnorm, x_norm, weight = ctx.saved_variables
        # d_input = d_weight = d_bias = d_minl = None
        
        # nsize = x.size(2) * x.size(3)
        # d_xnorm = grad * weight
        # d_x = inv_cnorm * (d_xnorm - (d_xnorm*x).sum([2,3],keepdim=True) * x_norm *isigma_grad(inv_cnorm, clip) / nsize)  
        # d_input = d_x - d_x.mean([2,3], keepdim=True)

        # if ctx.needs_input_grad[1]:
        #     d_weight = x_norm * grad
        # if ctx.needs_input_grad[2]:
        #     d_bias = grad

        inp, weight, mean, invstd, clip = ctx.saved_variables
        d_input, d_weight, d_bias = batch_backward(grad, inp, weight, mean, invstd, clip.item())
        return d_input, d_weight, d_bias, None, None



class batchCondNorm(nn.Module):
    def __init__(self, planes=64, minl=1.0, eps=1e-5):
        super().__init__()
        self.planes = planes
        self.minl = minl
        self.clip = torch.tensor(minl, dtype=torch.double)
        self.eps = eps
        self.weight = Parameter(torch.ones(planes))
        self.bias = Parameter(torch.zeros(planes))
    
    def forward(self, x):
        return batch_cond_norm.apply(x, self.weight, self.bias, self.clip, self.eps)

    def extra_repr(self):
        s = 'planes={planes}, minl={minl}, epsilon={eps}'
        return s.format(**self.__dict__)
        
