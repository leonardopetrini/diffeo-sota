from .transform import *
from .vgg import *
from .resnet import *

class build_Scattering(nn.Module):
    def __init__(self, args, in_channels=3, num_classes=10):
        super(ScatteringNet, self).__init__()

        if args.scattering_mode == 1:
            raise NotImplementedError
        else:
            raise NotImplementedError
            # K =

        if args.net == 'ResNet':
            model = Scattering2dResNet(K, args.width).to(args.device)