'''
resnet with cond norm, a new version with comparable structure.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch._jit_internal import weak_module, weak_script_method, List


# original resnets

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normlayer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = normlayer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), )
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out
        out = F.relu(out, )
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, normlayer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, )
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, )
        self.bn2 = normlayer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, )
        self.bn3 = normlayer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = F.relu(self.bn2(self.conv2(out)), inplace=False)
        out = self.bn3(self.conv3(out))
        out = self.shortcut(x) + out
        out = F.relu(out, inplace=False)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, nc=3, nfilters=64, normlayer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.in_planes = nfilters
        self.norm = normlayer
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, nfilters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nfilters*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nfilters*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nfilters*8, num_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(nfilters*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, normlayer=self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_conv(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def get_score(self, out):
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        # out = out.squeeze(-1).squeeze(-1)
        out = self.linear(out)
        return out

    def forward(self, x):
        out = self.get_conv(x)
        score = self.get_score(out)
        return score

    def compute_shape(self, shape):
        h, w = shape[0]//8, shape[1] // 8
        return h, w


def ResNet18(nclass=10, nc=1, normlayer=nn.BatchNorm2d):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=nclass, nc=nc, normlayer=normlayer)

def ResNet34(nclass=10, nc=1, normlayer=nn.BatchNorm2d):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=nclass, nc=nc, normlayer=normlayer)

def ResNet50(nclass=10, nc=1, normlayer=nn.BatchNorm2d):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=nclass, normlayer=normlayer)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
