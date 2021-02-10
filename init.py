import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *

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
    transforms_list = []

    if args.dataset == 'mnist':
        transforms_list.append(transforms.ToTensor())
        if args.diffeo:
            transforms_list.append(Diffeo(args.sT, args.rT, args.scut, args.rcut, args.cutmin, args.cutmax))
        transform_train = transforms.Compose(transforms_list)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST
    if args.dataset == 'fashionmnist':
        transforms_list.append(transforms.ToTensor())
        if args.diffeo:
            transforms_list.append(Diffeo(args.sT, args.rT, args.scut, args.rcut, args.cutmin, args.cutmax))
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
        transform_train = transforms.Compose(transforms_list)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.FashionMNIST
    if args.dataset == 'cifar10':
        if args.random_crop and not args.onlydiffeo:
            transforms_list.append(transforms.RandomCrop(32, padding=4))
        if args.hflip and not args.onlydiffeo:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        if args.diffeo:
            transforms_list.append(Diffeo(args.sT, args.rT, args.scut, args.rcut, args.cutmin, args.cutmax))
        transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(transforms_list)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10

    trainset = dataset(
        root='/home/lpetrini/data/' + args.dataset, train=True, download=True, transform=transform_train)
    testset = dataset(
        root='/home/lpetrini/data/' + args.dataset, train=False, download=True, transform=transform_test)


    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= 5) - 1
    P = len(trainset)
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
    net = None
    if args.net == 'VGG16':
        net = VGG('VGG16')
    if args.net == 'ResNet18':
        net = ResNet18(num_ch=num_ch, num_classes=num_classes)
    # if args.net == 'PreActResNet18':
    #     net = PreActResNet18()
    # if args.net == 'GoogLeNet':
    #     net = GoogLeNet()
    # if args.net == 'DenseNet121':
    #     net = DenseNet121()
    # if args.net == 'ResNeXt29_2x64d':
    #     net = ResNeXt29_2x64d()
    if args.net == 'MobileNetV2':
        net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
    # if args.net == 'DPN92':
    #     net = DPN92()
    # if args.net == 'ShuffleNetG2':
    #     net = ShuffleNetG2()
    # if args.net == 'SENet18':
    #     net = SENet18()
    # if args.net == 'ShuffleNetV2':
    #     net = ShuffleNetV2(1)
    if args.net == 'EfficientNetB0':
        net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
    # if args.net == 'RegNetX_200MF':
    #     net = RegNetX_200MF()
    # if args.net == 'SimpleDLA':
    #     net = SimpleDLA()
    if args.net == 'ConvNetL4':
        net = ConvNetL4()
    if args.net == 'FC':
        net = FC()
    assert net is not None, 'Network architecture not in the list!'
    net = net.to(args.device)
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return trainloader, testloader, net