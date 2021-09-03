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
        return self.fcn(x)


def DenseNetL1(num_ch=1, num_classes=10, h=64):
    return DenseNet(n_layers=1, input_dim=num_ch, h=h, out_dim=num_classes)

def DenseNetL2(num_ch=1, num_classes=10, h=64):
    return DenseNet(n_layers=2, input_dim=num_ch, h=h, out_dim=num_classes)

def DenseNetL4(num_ch=1, num_classes=10, h=64):
    return DenseNet(n_layers=4, input_dim=num_ch, h=h, out_dim=num_classes)

def DenseNetL6(num_ch=1, num_classes=10, h=64):
    return DenseNet(n_layers=6, input_dim=num_ch, h=h, out_dim=num_classes)