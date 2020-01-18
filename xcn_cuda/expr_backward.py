import torch
import bcn_cuda
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class BCN(torch.nn.Module):
    def __init__(self, planes=64, minl=1.0, eps=1e-5):
        super().__init__()
        self.planes = planes
        self.minl = minl
        self.eps = eps
        self.weight = Parameter(torch.ones(1,planes,1,1))
        self.bias = Parameter(torch.zeros(1,planes,1,1))
    
    def forward(self, x):
        # x: N x C x H x W,  4D Tensor
        x = x - x.mean([0,2,3], keepdim=True)
        
        norm = torch.sqrt(torch.mean(x**2, dim=[0,2,3], keepdim=True) + self.eps)
        inv_cnorm = 1 / torch.max(norm, self.minl*torch.ones_like(norm))
        
        xn = inv_cnorm * x
        
        return self.weight * xn + self.bias

x = torch.cat([torch.randn(16,32,8,8)*2, torch.rand(16,32,8,8)*0.5],dim=1).cuda()

m = BCN(64,1.0,1e-5).cuda()
a1 = Variable(x, requires_grad=True)
# a1.retain_grad()
y1 = m(a1)
y1.retain_grad()
z1 = y1.abs().sum()
z1.backward()

# print(y1.data[0,0,:4,:4])
# print(y1.grad[0,0,:4,:4])
# print(a1.grad[0,0,:4,:4])

weight = torch.ones(x.size(1)).cuda()
bias = torch.zeros(x.size(1)).cuda()
y2, mean, inv_std = bcn_cuda.forward(x,weight,bias, 1.0, 1e-5)
d_inp, d_weight, d_bias = bcn_cuda.backward(y1.grad, x, weight, mean, inv_std, 1.0)
# print(y2.data[0,0,:4,:4])
# print(d_inp[0,0,:4,:4])

print((a1.grad-d_inp).abs().max().item(), d_inp.abs().max())           # 5e-7 / 1.3x