'''FCC in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, h=500):
        super(FC, self).__init__()
        self.fc1   = nn.Linear(32*32*3, h)
        self.fc2   = nn.Linear(h, h)
        self.fc3   = nn.Linear(h, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# class FC(nn.Module):
#     def __init__(self, d=32*32*3, h=10000, c=10, L=1, act=torch.relu, bias=True, last_bias=True, var_bias=1):
#         super().__init__()
#
#         hh = d
#         for i in range(L):
#             W = torch.randn(h, hh, device='cuda')
#
#             setattr(self, "W{}".format(i), W)
#             if bias:
#                 self.register_parameter("B{}".format(i), nn.Parameter(torch.randn(h, device='cuda').mul(var_bias**0.5)))
#             hh = h
#
#         self.register_parameter("W{}".format(L), nn.Parameter(torch.randn(c, hh, device='cuda')))
#         if last_bias:
#             self.register_parameter("B{}".format(L), nn.Parameter(torch.randn(c, device='cuda').mul(var_bias**0.5)))
#
#         self.L = L
#         self.act = act
#         self.bias = bias
#         self.last_bias = last_bias
#
#     def forward(self, x):
#
#         x = x.view(x.size(0), -1)
#
#         for i in range(self.L + 1):
#             W = getattr(self, "W{}".format(i))
#
#             if isinstance(W, nn.ParameterList):
#                 W = torch.cat(list(W))
#
#             if self.bias and i < self.L:
#                 B = self.bias * getattr(self, "B{}".format(i))
#             elif self.last_bias and i == self.L:
#                 B = self.last_bias * getattr(self, "B{}".format(i))
#             else:
#                 B = 0
#
#             h = x.size(1)
#
#             if i < self.L:
#                 x = x @ (W.t() / h ** 0.5)
#                 x = self.act(x + B)
#             else:
#                 x = x @ (W.t() / h) + B
#
#         if x.shape[1] == 1:
#             return x.view(-1)
#         return x