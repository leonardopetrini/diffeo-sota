import torch

import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):

    def __init__(self, input_dim, h, last=False):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim, h))
        self.b = nn.Parameter(torch.randn(h))
        self.last = last

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        input_dim = x.size(1)
        if self.last:
            y = x @ self.w / input_dim
        else:
            y = x @ self.w / input_dim ** .5 + self.b
            y = F.relu(y)
        return y


class DenseNet(nn.Module):

    def __init__(self, n_layers, input_dim, h, out_dim):
        super().__init__()
        self.fcn = nn.Sequential(DenseBlock(input_dim, h),
                                 *[DenseBlock(h, h) for _ in range(n_layers - 2)],
                                 DenseBlock(h, out_dim, last=True))

    def forward(self, x):
        y = self.fcn(x)
        return y.view(-1)


class ConvBlock(nn.Module):

    def __init__(self, input_dim, h, filter_size, stride):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(h, input_dim, filter_size, filter_size))
        self.w2 = nn.Parameter(torch.randn(h, h, filter_size, filter_size))
        self.stride = stride

    def forward(self, x):
        filter_size = self.w1.size(-1)
        x = F.pad(x, (filter_size - 1, 0, filter_size - 1, 0), mode='circular')
        y = F.conv2d(x, self.w1 / filter_size, stride=self.stride)
        y = F.relu(y)
        y = F.pad(y, (filter_size - 1, 0, filter_size - 1, 0), mode='circular')
        y = F.conv2d(y, self.w2 / filter_size)
        y = F.relu(y)
        return y


class ConvNet(nn.Module):

    def __init__(self, n_blocks, input_dim, h, filter_size, stride, out_dim):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(input_dim, h, filter_size, stride),
                                  *[ConvBlock(h, h, filter_size, stride) for _ in range(n_blocks - 1)])
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        y = self.conv(x)
        y = torch.squeeze(F.avg_pool2d(y, kernel_size=y.size(-1)))
        h = self.beta.size(0)
        y = y @ (self.beta / h)
        return y.view(-1)


def ConvNetL4():
    n_blocks = int(4 / 2)
    return ConvNet(n_blocks=n_blocks, input_dim=3, h=64, filter_size=5, stride=2, out_dim=1)