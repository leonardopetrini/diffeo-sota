import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
from models.pretrained import *

import sys
import warnings
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
try:
    from transform import Diffeo
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot use diffeo-transform! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")


def init_fun(args):

    torch.manual_seed(args.seed_init)
    train_list = []
    test_list = [transforms.ToTensor()]

    diffeo_transform = Diffeo(args.sT, args.rT, args.scut, args.rcut, args.cutmin, args.cutmax)

    if args.dataset == 'mnist':
        train_list.append(transforms.ToTensor())
        if args.diffeo:
            train_list.append(diffeo_transform)
        dataset = torchvision.datasets.MNIST
    if args.dataset == 'fashionmnist':
        train_list.append(transforms.ToTensor())
        if args.diffeo:
            train_list.append(diffeo_transform)
        train_list.append(transforms.Normalize((0.5,), (0.5,)))
        test_list.append(transforms.Normalize((0.5,), (0.5,)))
        dataset = torchvision.datasets.FashionMNIST
    if args.dataset == 'cifar10':
        if args.random_crop and not args.onlydiffeo:
            train_list.append(transforms.RandomCrop(32, padding=4))
        if args.hflip and not args.onlydiffeo:
            train_list.append(transforms.RandomHorizontalFlip())
        train_list.append(transforms.ToTensor())
        if args.diffeo:
            train_list.append(diffeo_transform)
        train_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        test_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        dataset = torchvision.datasets.CIFAR10

    if args.pretrained:
        train_list.append(transforms.Resize((224, 224), interpolation=3))
        test_list.append(transforms.Resize((224, 224), interpolation=3))

    transform_train = transforms.Compose(train_list)
    transform_test = transforms.Compose(test_list)

    trainset = dataset(
        root='/home/lpetrini/data/' + args.dataset, train=True, download=True, transform=transform_train)
    testset = dataset(
        root='/home/lpetrini/data/' + args.dataset, train=False, download=True, transform=transform_test)


    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= 5) - 1
    P = len(trainset)
    if args.random_labels:
        trainset.targets = trainset.targets[torch.randperm(trainset.targets.nelement())]
    if args.ptr:
        # take random subset of training set
        perm = torch.randperm(P)
        trainset = torch.utils.data.Subset(trainset, perm[:args.ptr])

        # adjust number of epochs with cap at 5k
        args.epochs = min(int(args.epochs * P / args.ptr), 5000)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.loss == 'hinge':
        # change to binary labels
        testset.targets = 2 * (torch.as_tensor(testset.targets) >= 5) - 1
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    num_ch = 1 if 'mnist' in args.dataset else 3
    num_classes = 1 if args.loss == 'hinge' else 10
    imsize = 28 if 'mnist' in args.dataset else 32
    net = None
    if not args.pretrained:
        if 'VGG' in args.net:
            if 'bn' in args.net:
                bn = True
                net_name = args.net[:-2]
            else:
                bn = False
                net_name = args.net
            net = VGG(net_name, num_ch=num_ch, num_classes=num_classes, batch_norm=bn)
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
            net = LeNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MobileNetV2':
            net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'EfficientNetB0':
            net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ConvNetL4':
            net = ConvNetL4(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'DenseNetL2':
            net = DenseNetL2(num_ch=num_ch * imsize ** 2, num_classes=num_classes)
        if args.net == 'DenseNetL4':
            net = DenseNetL4(num_ch=num_ch * imsize ** 2, num_classes=num_classes)
        if args.net == 'DenseNetL6':
            net = DenseNetL6(num_ch=num_ch * imsize ** 2, num_classes=num_classes)
        if args.net == 'FC':
            net = FC()
        assert net is not None, 'Network architecture not in the list!'
        net = net.to(args.device)
    else:
        cfg.merge_from_file(f'./models/pretrained/configs/{args.net}.yaml')
        cfg.freeze()
        net = build_EfficientNet(cfg)

    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return trainloader, testloader, net