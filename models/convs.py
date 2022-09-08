import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockNT(nn.Module):
    """
        Convolutional block with NTK initialization
    """

    def __init__(self, input_ch, h, filter_size, stride, pbc, batch_norm=0):

        super().__init__()
        self.w = nn.Parameter(
            torch.randn(h, input_ch, filter_size, filter_size)
        )
        self.b = nn.Parameter(torch.randn(h))
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_features=h)
        self.batch_norm = batch_norm
        self.stride = stride
        self.pbc = pbc

    def forward(self, x):

        if self.pbc:
            pool_size = (self.w.size(-1) - 1) // 2
            x = F.pad(x, (pool_size, pool_size, pool_size,
                          pool_size), mode='circular')
        # if self.w.shape[1] > 1:
        h = self.w[0].numel()
        # else:
        #     h = 0.01
        x = F.conv2d(x, self.w / h ** .5,
                     bias=self.b, # / h ** .5,
                     stride=self.stride).relu()
        if self.batch_norm:
            x = self.bn(x)
        return x

class ConvBlockSD(nn.Module):
    """
    Convolutional block with standard torch initialization
    """

    def __init__(self, input_ch, h, filter_size, stride, pbc, bias=True, batch_norm=False):

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_ch,
            out_channels=h,
            kernel_size=filter_size,
            stride=stride,
            bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_features=h)
        self.batch_norm = batch_norm
        self.filter_size = filter_size
        self.pbc = pbc

    def forward(self, x):

        if self.pbc:
            pool_size = int((self.filter_size - 1) / 2.)
            x = F.pad(x, (pool_size, pool_size, pool_size,
                      pool_size), mode='circular')
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)

        return x

class ConvNetGAPMF(nn.Module):
    """
    Convolutional neural network with MF initialization and global average
    pooling
    """

    def __init__(
            self, n_blocks, input_ch, h,
            filter_size, stride, pbc, out_dim, batch_norm, last_bias=1):

        super().__init__()
        self.conv = nn.Sequential(ConvBlockNT(input_ch, h, filter_size, stride, pbc, batch_norm),
                                  *[ConvBlockNT(h, h, filter_size, stride if i % 2 == 1 else 1, pbc, batch_norm)
                                  # *[ConvBlockNT(h, h, filter_size, stride, pbc, batch_norm)
                                       for i in range(n_blocks-1)]
                                  )
        self.beta = nn.Parameter(torch.randn(h, out_dim))
        if last_bias:
            self.last_bias = nn.Parameter(torch.randn(1))
        else:
            self.last_bias = None

    def forward(self, x):

        y = self.conv(x)
        y = y.mean(dim=[-1, -2]) # .squeeze()
        y = y @ self.beta / self.beta.size(0)
        if self.last_bias is not None:
            y += self.last_bias
        return y
