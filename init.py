import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
from models.pretrained import *
from models.scattering import *
from fourier import *

import sys
import warnings
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
try:
    from transform import Diffeo
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot use diffeo-transform! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")

import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def init_fun(args):

    ### Define dataset and preprocessing ###

    torch.manual_seed(args.seed_init)
    train_list = []
    test_list = [transforms.ToTensor()]

    diffeo_transform = Diffeo(args.sT, args.rT, args.scut, args.rcut, args.cutmin, args.cutmax)
    filter_transform = LowHighPassFilter(args.filter_p)

    if 'imagenet' not in args.dataset:

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
            if args.train_filtered:
                train_list.append(filter_transform)

            train_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
            test_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

            if args.scattering_mode:
                train_list.append(ScatteringTransform(args))
                test_list.append(ScatteringTransform(args))

            if args.black_and_white:
                train_list.append(AvgChannels())
                test_list.append(AvgChannels())

            dataset = torchvision.datasets.CIFAR10

        if args.dataset == 'svhn':
            if args.random_crop and not args.onlydiffeo:
                train_list.append(transforms.RandomCrop(32, padding=4))
            train_list.append(transforms.ToTensor())
            if args.diffeo:
                train_list.append(diffeo_transform)
            if args.train_filtered:
                train_list.append(filter_transform)

            train_list.append(transforms.Normalize((0.4376, 0.4437, 0.4728), (0.1976, 0.2006, 0.1965)))
            test_list.append(transforms.Normalize((0.4523, 0.4524, 0.4689), (0.2190, 0.2261, 0.2279)))

            if args.black_and_white:
                train_list.append(AvgChannels())
                test_list.append(AvgChannels())

            dataset = torchvision.datasets.SVHN

        if args.pretrained:
            train_list.append(transforms.Resize((224, 224), interpolation=3))
            test_list.append(transforms.Resize((224, 224), interpolation=3))

        transform_train = transforms.Compose(train_list)
        transform_test = transforms.Compose(test_list)

        if args.dataset == 'svhn':
            train_arg = {'split': 'train'}
            test_arg = {'split': 'test'}
        else:
            train_arg = {'train': True}
            test_arg = {'train': False}

        trainset = dataset(
            root='/home/lpetrini/data/' + args.dataset, download=True, transform=transform_train, **train_arg)
        testset = dataset(
            root='/home/lpetrini/data/' + args.dataset, download=True, transform=transform_test, **test_arg)

        trainset.targets = trainset.labels
        testset.targets = testset.labels

        imsize = 28 if 'mnist' in args.dataset else 32

    else:
        assert 'tiny-imagenet' in args.dataset, 'Only tiny version of imagenet is implemented!'
        imsize = int(args.dataset[-2:])

        if args.random_crop and not args.onlydiffeo:
            train_list.append(transforms.RandomCrop(64, padding=4))
        if args.hflip and not args.onlydiffeo:
            train_list.append(transforms.RandomHorizontalFlip())
        train_list.append(transforms.ToTensor())
        if args.diffeo:
            train_list.append(diffeo_transform)
        if args.train_filtered:
            train_list.append(filter_transform)

        train_list.append(transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2768, 0.2688, 0.2818)))
        test_list.append(transforms.Normalize((0.4824, 0.4495, 0.3981), (0.2768, 0.2691, 0.2827)))

        if imsize != 64 and not args.pretrained:
            train_list.append(transforms.Resize((imsize, imsize), interpolation=3))
            test_list.append(transforms.Resize((imsize, imsize), interpolation=3))

        if args.scattering_mode:
            train_list.append(ScatteringTransform(args))
            test_list.append(ScatteringTransform(args))

        if args.black_and_white:
            train_list.append(AvgChannels())
            test_list.append(AvgChannels())

        if args.pretrained:
            train_list.append(transforms.Resize((224, 224), interpolation=3))
            test_list.append(transforms.Resize((224, 224), interpolation=3))

        transform_train = transforms.Compose(train_list)
        transform_test = transforms.Compose(test_list)

        trainset = torchvision.datasets.ImageFolder('/home/lpetrini/data/tiny-imagenet-200/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder('/home/lpetrini/data/tiny-imagenet-200/val', transform=transform_test)

    # number of classes
    nc = max(trainset.targets) + 1

    ## Build trainloader ##
    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= nc // 2) - 1
    P = len(trainset)
    if args.random_labels:
        trainset.targets = trainset.targets[torch.randperm(trainset.targets.nelement())]
    if args.ptr:
        # take random subset of training set
        perm = torch.randperm(P)
        trainset = torch.utils.data.Subset(trainset, perm[:args.ptr])

        if args.rescale_epochs:
            # adjust number of epochs with cap at 5k
            args.epochs = min(int(args.epochs * P / args.ptr), 5000)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    ## Build testloader ##
    if args.loss == 'hinge':
        # change to binary labels
        testset.targets = 2 * (torch.as_tensor(testset.targets) >= nc // 2) - 1
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    num_ch = 1 if 'mnist' in args.dataset or args.black_and_white else 3
    num_classes = 1 if args.loss == 'hinge' else nc

    ### Define network architecture ###

    net = None
    if not args.pretrained and not args.scattering_mode:
        if 'VGG' in args.net:
            if 'bn' in args.net:
                bn = True
                net_name = args.net[:-2]
            else:
                bn = False
                net_name = args.net
            net = VGG(net_name, num_ch=num_ch, num_classes=num_classes, batch_norm=bn, param_list=args.param_list)
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
        if args.net == 'GoogLeNet':
            net = GoogLeNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MobileNetV2':
            net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'DenseNet121':
            net = DenseNet121(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'EfficientNetB0':
            net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
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
    elif args.pretrained:
        cfg.merge_from_file(f'./models/pretrained/configs/{args.net}.yaml')
        cfg.freeze()
        net = build_EfficientNet(cfg)
    else:
        raise NotImplementedError
        # net = build_Scattering(args)

    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return trainloader, testloader, net


class AvgChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be averaged.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return tensor.mean(dim=-3, keepdim=True)

    def __repr__(self):
        return self.__class__.__name__