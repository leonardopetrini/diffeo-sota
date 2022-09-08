import torch.backends.cudnn as cudnn

from .alexnet import *
from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *
from .fc import *
from .simplenets import *
# from .convnext import *
from .convs import *
# from .pretrained import *
# from .scattering import *


def model_initialization(args, image_size, num_classes):
    """
    Neural netowrk initialization.
    :param args: parser arguments
    :param image_size: input image size (needed for FC nets)
    :param num_classes: number of outputs
    :return: neural network as torch.nn.Module
    """

    if args.ch == 0:
        num_ch = 1 if 'mnist' in args.dataset or args.black_and_white or 'twopoints' in args.dataset else 3
    else:
        num_ch = args.ch

    num_classes = 1 if args.loss == 'hinge' else num_classes

    ### Define network architecture ###
    if args.seed_net != -1:
        torch.manual_seed(args.seed_net)
    net = None
    if not args.pretrained: # and not args.scattering_mode
        if 'VGG' in args.net:
            if 'bn' in args.net:
                bn = True
                net_name = args.net[:-2]
            else:
                bn = False
                net_name = args.net
            net = VGG(
                net_name,
                num_ch=num_ch,
                num_classes=num_classes,
                batch_norm=bn,
                param_list=args.param_list,
                pooling=args.pooling,
                width_factor=args.width_factor,
                stride=args.stride)
        if args.net == 'AlexNet':
            net = AlexNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet18':
            net = ResNet18(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet34':
            net = ResNet34(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet50':
            net = ResNet50(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet101':
            net = ResNet101(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'LeNet':
            net = LeNet(num_ch=num_ch, num_classes=num_classes, stride=args.stride)
        if args.net == 'GoogLeNet':
            net = GoogLeNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MobileNetV2':
            net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'DenseNet121':
            net = DenseNet121(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'EfficientNetB0':
            net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ConvNextT':
            net = ConvNextT(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MinCNN':
            net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size, param_list=args.param_list)
        if args.net == 'LCN':
            net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size)
        if args.net == 'DenseNetL2':
            net = DenseNetL2(num_ch=num_ch * image_size ** 2, num_classes=num_classes, h=args.width)
        if args.net == 'DenseNetL4':
            net = DenseNetL4(num_ch=num_ch * image_size ** 2, num_classes=num_classes, h=args.width)
        if args.net == 'DenseNetL6':
            net = DenseNetL6(num_ch=num_ch * image_size ** 2, num_classes=num_classes, h=args.width)
        if args.net == 'ConvGAP':
            net = ConvNetGAPMF(n_blocks=args.depth, input_ch=num_ch, h=args.width, filter_size=args.filter_size,
                               stride=args.stride, pbc=args.pbc, out_dim=num_classes, batch_norm=args.batch_norm)
        if args.net == 'FC':
            net = FC()
        if args.net == 'ScatteringLinear':
            net = ScatteringLinear(n=image_size, ch=num_ch, J=args.J, L=args.L, num_classes=num_classes)
        assert net is not None, 'Network architecture not in the list!'
        net = net.to(args.device)
    elif args.pretrained:
        cfg.merge_from_file(f'./models/pretrained/configs/{args.net}.yaml')
        cfg.freeze()
        net = build_EfficientNet(cfg)
    else:
        raise NotImplementedError

    if args.random_features:
        for param in [p for p in net.parameters()][:-2]:
            param.requires_grad = False

    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net