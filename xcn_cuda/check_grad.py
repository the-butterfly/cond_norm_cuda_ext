# check grad in three methods, 
# Correct !

import torch                                                                                                                                                                              
import cond_norm_baseline as cnb
import cond_norm as cn            
from torch.autograd import Variable
import torch.nn.functional as F

@torch.jit.script
def sign_grad(x:torch.Tensor):
    return (x >= 0).type_as(x) * torch.ones_like(x)

H = 4; W = 4
inp = torch.rand(1,4,H,W) * 2           # 2 / 4
# inp = torch.load('mnist_sample.pt')[None].double()

# inference
clip = torch.tensor(1)
a1 = Variable(inp, requires_grad=True)
x = a1 - a1.mean([2,3],keepdim=True)
x.retain_grad()
norm = torch.sqrt(torch.mean(x**2,dim=[2,3], keepdim=True)+1e-5)
norm.retain_grad()
inv_cnorm = 1 / torch.clamp_min(norm, clip)
inv_cnorm.retain_grad()
x_norm = x * inv_cnorm
x_norm.retain_grad()
c1 = x_norm.abs().sum()
c1.retain_grad()
c1.backward()

# instance norm
a2 = Variable(inp, requires_grad=True)
m = torch.nn.InstanceNorm2d(4)
c2 = m(a2).abs().sum()
c2.backward()

# calculate gradients
d_xnorm = x_norm.grad # * weight
d_inv = (d_xnorm * x.data).sum([2,3],keepdim=True)
d_norm = d_inv * (-inv_cnorm.data**2) * sign_grad(norm.data-clip.data)

# d_d_x = d_norm * x.data / norm.data / (H*W)
d_d_x = d_norm * x.data * inv_cnorm.data / (H*W)          # equal

# d_x = d_xnorm * inv_cnorm.data + d_d_x * x.data
d_x = d_xnorm * inv_cnorm.data + d_d_x                    # correct
# 1. original 
# d_input = d_x - d_x.mean([2,3], keepdim=True)
# 2. simplify  note: d_x.mean == g_out.mean() * w * inv_sigma ? YES !
d_input = d_x - d_xnorm.mean([2,3], keepdim=True) * inv_cnorm.data

# d_x_float = d_xnorm.float() * inv_cnorm.data.float() + d_norm.float() * norm.data.float() * x.data.float()**2 / (H*W)
# (d_x_float.double() - x.grad).abs().max()           # err: 2e-8

print("loss: Customized / Instance Norm")
print(c1.item(), c2.item())
print("grad diff by element(max):")
print("AutoDiff - InstNorm", (a1.grad - a2.grad).abs().max().item())
print("HandGrad - AutoDiff", (d_input - a1.grad).abs().max().item())
print("HandGrad - InstNorm", (d_input - a2.grad).abs().max().item())
print("grad maximum:  hand grad / inst norm / auto diff")
print(a1.grad.abs().max().item(), a2.grad.abs().max().item(), d_input.abs().max().item())
# print(x.data.abs().max().item())

