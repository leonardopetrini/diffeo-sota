import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockNT(nn.Module):
    """
        Convolutional block with NTK initialization
    """

    def __init__(self, input_ch, h, filter_size, stride):

        super().__init__()
        self.w = nn.Parameter(
            torch.randn(h, input_ch, filter_size, filter_size)
        )
        self.b = nn.Parameter(torch.randn(h))
        self.stride = stride

    def forward(self, x):

        pool_size = (self.w.size(-1) - 1) // 2
        y = F.pad(x, (pool_size, pool_size, pool_size,
                      pool_size), mode='circular')
        h = self.w[0].numel()
        y = F.conv2d(y, self.w / h ** .5,
                     bias=self.b / h ** .5,
                     stride=self.stride)
        y = F.relu(y)

        return y


class ConvBlockSD(nn.Module):
    """
    Convolutional block with standard torch initialization
    """

    def __init__(self, input_ch, h, filter_size, stride, bias=True, batch_norm=False):

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

    def forward(self, x):

        pool_size = int((self.filter_size - 1) / 2.)
        y = F.pad(x, (pool_size, pool_size, pool_size,
                      pool_size), mode='circular')
        y = self.conv(y)
        if self.batch_norm:
            y = self.bn(y)
        y = F.relu(y)

        return y


class ConvNetDenseChMF(nn.Module):
    '''
    Convolutional neural network with MF initialization and vectorialization
    (after the last convolutional layer, a fully connected layer is applied
    to all output channels mixing them)
    '''

    def __init__(
            self, n_blocks, input_dim, input_ch,
            h, filter_size, stride, out_dim):

        super().__init__()
        self.conv = nn.Sequential(ConvBlockNT(input_ch, h, filter_size, stride),
                                  *[ConvBlockNT(h, h, filter_size, stride)
                                      for _ in range(n_blocks-1)]
                                  )
        self.beta2 = nn.Parameter(torch.randn(h, input_dim))
        self.beta1 = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):

        y = self.conv(x)
        y = y.reshape(y.size(0), y.size(1), -1)
        y = self.beta2 * y / y.size(-1) ** .5
        y = y.sum(-1)
        y = F.relu(y)
        y = y @ self.beta1 / self.beta1.size(0)

        return y


class ConvNetDenseFullMF(nn.Module):
    '''
    Convolutional neural network with MF initialization and vectorialization
    (after the last convolutional layer, a fully connected layer is applied
    to each channel separately)
    '''

    def __init__(
            self, n_blocks, input_dim, input_ch,
            h, filter_size, stride, out_dim):

        super().__init__()
        self.conv = nn.Sequential(ConvBlockNT(input_ch, h, filter_size, stride),
                                  *[ConvBlockNT(h, h, filter_size, stride)
                                      for _ in range(n_blocks-1)]
                                  )
        self.beta2 = nn.Parameter(torch.randn(h * input_dim, h))
        self.beta1 = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):

        y = self.conv(x)
        y = y.reshape(y.size(0), -1)
        y = y @ self.beta2 / self.beta2.size(0) ** .5
        y = F.relu(y)
        y = y @ self.beta1 / self.beta1.size(0)

        return y


class ConvNetLinMF(nn.Module):
    '''
    Convolutional neural network with MF initialization and a linear layer
    after the convolutional ones.
    '''

    def __init__(
            self, n_blocks, input_dim, input_ch,
            h, filter_size, stride, out_dim):

        super().__init__()
        self.conv = nn.Sequential(ConvBlockNT(input_ch, h, filter_size, stride),
                                  *[ConvBlockNT(h, h, filter_size, stride)
                                      for _ in range(n_blocks-1)]
                                  )
        self.beta = nn.Parameter(torch.randn(h * input_dim ** 2, out_dim))

    def forward(self, x):

        y = self.conv(x)
        y = y.reshape(y.size(0), -1)
        y = y @ self.beta / self.beta.size(0)

        return y


class ConvNetGAPMF(nn.Module):
    """
    Convolutional neural network with MF initialization and global average
    pooling
    """

    def __init__(
            self, n_blocks, input_ch, h,
            filter_size, stride, out_dim):

        super().__init__()
        self.conv = nn.Sequential(ConvBlockNT(input_ch, h, filter_size, stride),
                                  *[ConvBlockNT(h, h, filter_size, stride)
                                      for _ in range(n_blocks-1)]
                                  )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):

        y = self.conv(x)
        y = y.mean(dim=[-1, -2]).squeeze()
        y = y @ self.beta / self.beta.size(0)

        return y


class ConvNetLinNT(nn.Module):
    '''
    Convolutional neural network with NTK initialization and a linear layer
    after the convolutional ones.
    '''

    def __init__(
            self, n_blocks, input_dim, input_ch,
            h, filter_size, stride, out_dim):

        super().__init__()
        self.conv = nn.Sequential(ConvBlockNT(input_ch, h, filter_size, stride),
                                  *[ConvBlockNT(h, h, filter_size, stride)
                                      for _ in range(n_blocks-1)]
                                  )
        self.beta = nn.Parameter(torch.randn(h * input_dim, out_dim))

    def forward(self, x):

        y = self.conv(x)
        y = y.reshape(y.size(0), -1)
        y = y @ self.beta / self.beta.size(0) ** .5

        return y


class ConvNetLinSD(nn.Module):
    '''
    Convolutional neural network with standard initialization and a linear layer
    after the convolutional ones.
    '''

    def __init__(
            self, n_blocks, input_dim, input_ch, h,
            filter_size, stride, out_dim, bias=True, batch_norm=False):

        super().__init__()
        self.convs = nn.Sequential(ConvBlockSD(
            input_ch, h, filter_size, stride,
            bias, batch_norm),
            *[ConvBlockSD(h, h, filter_size, stride,
                          bias, batch_norm)
              for _ in range(n_blocks-1)]
        )
        self.lin = nn.Linear(
            in_features=h * input_dim,
            out_features=out_dim,
            bias=False)

    def forward(self, x):

        y = self.convs(x)
        y = y.reshape(y.size(0), -1)
        y = self.lin(y)

        return y
