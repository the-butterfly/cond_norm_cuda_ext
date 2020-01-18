# 
# refer: https://github.com/facebookresearch/ImageNet-Adversarial-Training

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch._jit_internal import weak_module, weak_script_method, List


def denoising(name, l, embed=True, softmax=True):
    """
    Feature Denoising, Fig 4 & 5.
    """
    with tf.variable_scope(name):
        f = non_local_op(l, embed=embed, softmax=softmax)
        f = Conv2D('conv', f, l.shape[1], 1, strides=1, activation=get_bn(zero_init=True))
        l = l + f
    return l


def non_local_op(l, embed, softmax):
    """
    Feature Denoising, Sec 4.2 & Fig 5.
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = Conv2D('embedding_theta', l, n_in / 2, 1,
                       strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        phi = Conv2D('embedding_phi', l, n_in / 2, 1,
                     strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        g = l
    else:
        theta, phi, g = l, l, l
    if n_in > H * W or softmax:
        f = tf.einsum('niab,nicd->nabcd', theta, phi)
        if softmax:
            orig_shape = tf.shape(f)
            f = tf.reshape(f, [-1, H * W, H * W])
            f = f / tf.sqrt(tf.cast(theta.shape[1], theta.dtype))
            f = tf.nn.softmax(f)
            f = tf.reshape(f, orig_shape)
        f = tf.einsum('nabcd,nicd->niab', f, g)
    else:
        f = tf.einsum('nihw,njhw->nij', phi, g)
        f = tf.einsum('nij,nihw->njhw', f, theta)
    if not softmax:
        f = f / tf.cast(H * W, f.dtype)
    return tf.reshape(f, tf.shape(l))


class NonLocalMeans(nn.Module):
    def __init__(self, n_in, softmax=True):
        """
        Feature Denoising, Sec 4.2 & Fig 5.
        Args:
            embed (bool): whether to use embedding on theta & phi
            softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
        """
        super().__init__()
        self.in_ch = n_in
        self.theta = nn.Conv2d(n_in, n_in // 2, 1)
        self.phi = nn.Conv2d(n_in, n_in // 2, 1)
        self.softmax = softmax
        self.reset_params()

    def reset_params(self):
        self.theta.weight.data.normal_(0, 0.01)
        self.phi.weight.data.normal_(0, 0.01)

    def forward(self, x):
        theta = self.theta(x)
        phi = self.phi(x)
        n_in, H, W = x.shape[1:]
        if n_in > H*W or self.softmax:
            ff = torch.einsum('niab,nicd->nabcd', theta, phi)
            if self.softmax:
                orig_shape = ff.shape
                ff = ff.view(-1, H*W, H*W) / math.sqrt(self.in_ch)
                ff = torch.softmax(ff, dim=-1)
                ff = ff.reshape(orig_shape)
            ff = torch.einsum('nabcd,nicd->niab', ff, x)
        else:
            ff = torch.einsum('nihw,njhw->nij', phi, x)
            ff = torch.einsum('nij,nihw->njhw', ff, theta)
        if not self.softmax:
            ff = ff / (H * W)
        return ff.reshape_as(x)

class Denoising(nn.Module):
    def __init__(self, n_in, zero_init=True):
        super().__init__()
        self.nlm = NonLocalMeans(n_in)
        self.conv = nn.Conv2d(n_in, n_in, kernel_size=1)
        self.bn = nn.BatchNorm2d(n_in)
        if zero_init:
            self.bn.weight.data.fill_(0)

    def forward(self, x):
        y = self.bn(self.conv(self.nlm(x)))
        x = x + y
        return x