from .transform import *
from .vgg import *
from .resnet import *

from torch import nn

class ScatteringLinear(nn.Module):
    def __init__(self, n, ch=3, J=2, L=8, max_order=2, num_classes=10):
        super(ScatteringLinear, self).__init__()

        self.scattering = Scattering2D(J=J, shape=(n, n), L=L, max_order=max_order)
        nout = 8 if n == 32 else 7
        self.linear = nn.Linear(ch * 81 * nout * nout, num_classes)

    def forward(self, x):
        B = x.shape[0]
        s = self.scattering(x)
        return self.linear(s.view(B, -1))


# class build_Scattering(nn.Module):
#     def __init__(self, args, in_channels=3, num_classes=10):
#         super(ScatteringNet, self).__init__()
#
#         if args.scattering_mode == 1:
#             raise NotImplementedError
#         else:
#             raise NotImplementedError
#             # K =
#
#         if args.net == 'ResNet':
#             model = Scattering2dResNet(K, args.width).to(args.device)