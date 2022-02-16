import sys
import warnings

from .custom_transforms import *
from .diffeodataset import *
try:
    from .twopoints import *
except ModuleNotFoundError:
    warnings.warn('TwoPoints dataset is missing, work in progress...')

sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
try:
    from transform import Diffeo
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot use diffeo-transform! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")


def dataset_initialization(args):
    """
    Initialize train and test loaders for chosen dataset and transforms.
    :param args: parser arguments (see main.py)
    :return: trainloader, testloader, image size, number of classes.
    """

    if args.seed_data != -1:
        torch.manual_seed(args.seed_data)

    train_list = []
    test_list = [transforms.ToTensor()]

    diffeo_transform = Diffeo(args.sT, args.rT, args.scut, args.rcut, args.cutmin, args.cutmax)
    filter_transform = LowHighPassFilter(args.filter_p)

    if 'imagenet' not in args.dataset and 'diffeo' not in args.dataset and args.dataset != 'twopoints':

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
            if args.random_crop:
                train_list.append(transforms.RandomCrop(32, padding=4))
            if args.hflip:
                train_list.append(transforms.RandomHorizontalFlip())
            train_list.append(transforms.ToTensor())
            if args.diffeo:
                train_list.append(diffeo_transform)
            if args.train_filtered:
                train_list.append(filter_transform)

            train_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
            test_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

            if args.black_and_white:
                train_list.append(AvgChannels())
                test_list.append(AvgChannels())

            dataset = torchvision.datasets.CIFAR10

        if args.dataset == 'svhn':
            if args.random_crop:
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

        if args.gaussian_corruption_std:
            if not args.corruption_subspace_dimension:
                train_list.append(GaussianNoiseCorruption(std=args.gaussian_corruption_std))
            else:
                if args.dataset in ['cifar10' or 'svhn']:
                    numel = 32 ** 2 * 3
                elif 'mnist' in args.dataset:
                    numel = 28 ** 2
                train_list.append(RandomSubspaceCorruption(deff=args.corruption_subspace_dimension, d=numel,
                                                           std=args.gaussian_corruption_std))

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

        if dataset == 'svhn':
            trainset.targets = trainset.labels
            testset.targets = testset.labels

        imsize = 28 if 'mnist' in args.dataset else 32

    elif 'imagenet' in args.dataset:
        assert 'tiny-imagenet' in args.dataset, 'Only tiny version of imagenet is implemented!'
        imsize = int(args.dataset[-2:])

        if args.random_crop:
            train_list.append(transforms.RandomCrop(64, padding=4))
        if args.hflip:
            train_list.append(transforms.RandomHorizontalFlip())
        train_list.append(transforms.ToTensor())
        if args.diffeo:
            train_list.append(diffeo_transform)
        if args.train_filtered:
            train_list.append(filter_transform)

        train_list.append(transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2768, 0.2688, 0.2818)))
        test_list.append(transforms.Normalize((0.4824, 0.4495, 0.3981), (0.2768, 0.2691, 0.2827)))

        if imsize != 64 and not args.pretrained:
            train_list.append(transforms.Resize((imsize, imsize), interpolation=2))
            test_list.append(transforms.Resize((imsize, imsize), interpolation=2))

        if args.black_and_white:
            train_list.append(AvgChannels())
            test_list.append(AvgChannels())

        if args.pretrained:
            train_list.append(transforms.Resize((224, 224), interpolation=3))
            test_list.append(transforms.Resize((224, 224), interpolation=3))

        transform_train = transforms.Compose(train_list)
        transform_test = transforms.Compose(test_list)

        if args.group_tiny_classes:
            # group the 200 tiny-imagenet classes into 10 macro classes
            # the txt file has mappings from tiny-imagenet classes to macrolabels.
            # It assumes tiny classes are sorted in alphabetical order wrt classes ids (e.g. n0******)
            with open('/home/lpetrini/data/tiny-imagenet-200/tiny2macro.txt') as f:
                global t2m
                t2m = list(map(int, f.read().split('\n')))
            target_transform = lambda y: t2m[y]
        else:
            target_transform = lambda y: None

        trainset = torchvision.datasets.ImageFolder(
            '/home/lpetrini/data/tiny-imagenet-200/train', transform=transform_train, target_transform=target_transform)
        testset = torchvision.datasets.ImageFolder(
            '/home/lpetrini/data/tiny-imagenet-200/val', transform=transform_test, target_transform=target_transform)

    elif 'diffeo' in args.dataset:
        imsize = 32
        trainset, testset = torch.utils.data.random_split(
            diffeo_dataset(load_cifar_samples(5)[0][4], p=args.ptr + args.pte,
                                    cut=3, T=args.T, t=args.t),
            [args.ptr, args.pte])
        trainset.targets = trainset.dataset.tensors[1].tolist()
        testset.targets = testset.dataset.tensors[1].tolist()

    elif args.dataset == 'twopoints':
        imsize = 28
        if args.random_crop or args.hflip or args.diffeo:
            raise NotImplementedError
        else:
            transform = None

        trainset = TwoPointsDataset(xi=args.xi, d=imsize, gap=args.gap, pbc=args.pbc, norm=args.norm, train=True,
                                             transform=transform, testsize=args.ptr * 4)
        testset = TwoPointsDataset(xi=args.xi, d=imsize, gap=args.gap, pbc=args.pbc, norm=args.norm, train=False,
                                            transform=transform, testsize=args.ptr * 4)

    else:
        raise ValueError('`dataset` argument is invalid!')

    # number of classes
    num_classes = max(trainset.targets) + 1
    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= num_classes // 2) - 1
        testset.targets = 2 * (torch.as_tensor(testset.targets) >= num_classes // 2) - 1

    ## Build trainloader ##

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
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ## Build testloader ##

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    return trainloader, testloader, imsize, num_classes
