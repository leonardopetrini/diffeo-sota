from torch import nn
from kymatio.torch import Scattering2D


# class ScatteringTransform(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         n = 32 if args.dataset == 'cifar10' else 28
#         if args.scattering_mode == 1:
#             scattering = Scattering2D(J=args.J, L=args.L, shape=(n, n), max_order=1)
#         else:
#             scattering = Scattering2D(J=args.J, L=args.L, shape=(n, n))
#         self.scattering = scattering # .to(args.device)
#
#     def forward(self, x):
#         return self.scattering(x)
#
#     def __repr__(self):
#         return self.__class__.__name__ + f'(Scattering Transform)'