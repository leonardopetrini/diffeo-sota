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

    def __init__(self, n_blocks, input_dim, h, filter_size, stride, out_dim, act=None):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(input_dim, h, filter_size, stride),
                                  *[ConvBlock(h, h, filter_size, stride) for _ in range(n_blocks - 1)])
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        y = self.conv(x)
        y = torch.squeeze(F.avg_pool2d(y, kernel_size=y.size(-1)))
        h = self.beta.size(0)
        y = y @ (self.beta / h)
        return y


# class ConvBlock(nn.Module):
#
#     def __init__(self, input_dim, h, filter_size, stride, act):
#         super().__init__()
#         self.w1 = nn.Parameter(torch.randn(h, input_dim, filter_size, filter_size))
#         self.w2 = nn.Parameter(torch.randn(h, h, filter_size, filter_size))
#         self.stride = stride
#         self.act = act
#
#     def forward(self, x):
#         filter_size = self.w1.size(-1)
#         x = F.pad(x, (filter_size-1,0,filter_size-1,0), mode='circular')
#         h = self.w1[0].numel()
#         y = F.conv2d(x, self.w1 / h ** 0.5, stride=self.stride)
#         y = self.act(y)
#         y = F.pad(y, (filter_size-1,0,filter_size-1,0), mode='circular')
#         h = self.w2[0].numel()
#         y = F.conv2d(y, self.w2 / h ** 0.5)
#         y = self.act(y)
#         return y
#
# class ConvNet(nn.Module):
#
#     def __init__(self, n_blocks, input_dim, h, filter_size, stride, out_dim, act):
#         super().__init__()
#         self.conv = nn.Sequential(ConvBlock(input_dim, h, filter_size, stride, act),
#                                     *[ConvBlock(h, h, filter_size, stride, act) for _ in range(n_blocks-1)])
#         self.beta = nn.Parameter(torch.randn(h, out_dim))
#
#     def forward(self, x):
#         y = self.conv(x)
#         y = torch.squeeze(F.avg_pool2d(y, kernel_size=y.size(-1)))
#         h = self.beta.size(0)
#         y = y @ self.beta / h
#         return y


def ConvNetL4(num_ch=1, num_classes=10):
    n_blocks = int(4 / 2)
    return ConvNet(n_blocks=n_blocks, input_dim=num_ch, h=64, filter_size=5, stride=2, out_dim=num_classes, act=F.relu)


def DenseNetL2(num_ch=1, num_classes=10):
    return DenseNet(n_layers=2, input_dim=num_ch, h=64, out_dim=num_classes)

def DenseNetL4(num_ch=1, num_classes=10):
    return DenseNet(n_layers=4, input_dim=num_ch, h=64, out_dim=num_classes)

def DenseNetL6(num_ch=1, num_classes=10):
    return DenseNet(n_layers=6, input_dim=num_ch, h=64, out_dim=num_classes)