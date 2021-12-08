import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


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


import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class DenseBlock(nn.Module):
    def __init__(self, input_dim, h, last=False, dropout=False):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim, h))
        self.b = nn.Parameter(torch.randn(h))
        self.last = last
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        input_dim = x.size(1)
        if self.last:
            y = x @ self.w / input_dim
        else:
            y = x @ self.w / input_dim ** 0.5 + self.b
        if self.dropout is not None:
            y = self.dropout(y)
        return y if self.last else F.relu(y)


class DenseNet(nn.Module):
    def __init__(self, n_layers, input_dim, h, out_dim, dropout):
        super().__init__()
        self.fcn = nn.Sequential(
            DenseBlock(input_dim, h, dropout=dropout),
            *[DenseBlock(h, h, dropout=dropout) for _ in range(n_layers - 2)],
            DenseBlock(h, out_dim, last=True, dropout=dropout)
        )

    def forward(self, x):
        return self.fcn(x)


def DenseNetL1(num_ch=1, num_classes=10, h=64, dropout=False):
    return DenseNet(
        n_layers=1, input_dim=num_ch, h=h, out_dim=num_classes, dropout=dropout
    )


def DenseNetL2(num_ch=1, num_classes=10, h=64, dropout=False):
    return DenseNet(
        n_layers=2, input_dim=num_ch, h=h, out_dim=num_classes, dropout=dropout
    )


def DenseNetL4(num_ch=1, num_classes=10, h=64, dropout=False):
    return DenseNet(
        n_layers=4, input_dim=num_ch, h=h, out_dim=num_classes, dropout=dropout
    )


def DenseNetL6(num_ch=1, num_classes=10, h=64, dropout=False):
    return DenseNet(
        n_layers=6, input_dim=num_ch, h=h, out_dim=num_classes, dropout=dropout
    )


class MinCNN(nn.Module):
    def __init__(self, num_ch=3, num_classes=10, h=64, fs=5, ps=4, param_list=False):
        super(MinCNN, self).__init__()
        imsize = 32 if num_ch == 3 else 28
        if not param_list:
            self.conv = nn.Conv2d(num_ch, h, fs)
        else:
            self.conv = Conv2dList(num_ch, h, fs)
        self.fc = nn.Linear(((imsize - fs + 1) // ps) ** 2 * h, num_classes)
        self.ps = ps

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = F.max_pool2d(out, self.ps)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LocallyConnected2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride, bias=False
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size ** 2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class LCN(nn.Module):
    def __init__(self, num_ch=3, num_classes=10, h=64, fs=4, ps=4):
        super(LCN, self).__init__()
        imsize = 32 if num_ch == 3 else 28
        pad_size = fs - imsize % fs
        output_size = math.ceil(imsize / fs)
        self.pad = nn.ZeroPad2d((pad_size, 0, pad_size, 0))
        self.lc = LocallyConnected2d(in_channels=num_ch, out_channels=h, output_size=output_size, kernel_size=fs, stride=fs, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=ps, stride=ps)
        self.fc = nn.Linear((output_size // ps) ** 2 * h, num_classes)

    def forward(self, x):
        out = F.relu(self.lc(x))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    
class Conv2dList(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='constant'):
        super().__init__()

        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if bias is not None:
            bias = nn.Parameter(torch.empty(out_channels, ))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        n = max(1, 128 * 256 // (in_channels * kernel_size * kernel_size))
        weight = nn.ParameterList([nn.Parameter(weight[j: j + n]) for j in range(0, len(weight), n)])

        setattr(self, 'weight', weight)
        setattr(self, 'bias', bias)

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride

    def forward(self, x):

        weight = self.weight
        if isinstance(weight, nn.ParameterList):
            weight = torch.cat(list(self.weight))

        return F.conv2d(x, weight, self.bias, self.stride, self.padding)
