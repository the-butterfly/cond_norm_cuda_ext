import torch
from xcn_cuda import *
x = torch.rand(4,3,8,8).cuda()
w = torch.rand(3).cuda()
b = torch.rand(3).cuda()
from torch.autograd import Variable
a1 = Variable(x, requires_grad=True)
a2 = Variable(x, requires_grad=True)
o1, m1, s1 = instance_forward(x, w, b, 0.0, 1e-5)

mod = torch.nn.InstanceNorm2d(4, affine=True).cuda()
mod.weight.data = w.data
mod.bias.data = b.data
o2 = mod(a2)
o2.retain_grad()
print("output: ", (o2-o1).abs().max().item())
o2.abs().sum().backward()

gin, gw, gb = instance_backward(o2.grad.data, x, w, m1, s1, 0.0)
print("grad input: ", (gin-a2.grad.data).abs().max().item())
print("grad weight: ", (gw-mod.weight.grad.data).abs().max().item())
print("grad bias: ", (gb-mod.bias.grad.data).abs().max().item())
