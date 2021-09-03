import sys

import torchvision
from torchvision import transforms
import numpy as np

try:
    sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
except ModuleNotFoundError:
    print("Diffeo Module not found !!! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")
from diff import *
from transform import *
import image
from image import *
from etc import *
from tqdm.auto import tqdm
import glob
import matplotlib.pyplot as plt

from models import *
from models.pretrained import *


def typical_temperature(delta, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return 4 * delta ** 2 / (math.pi * n ** 2 * log)


def load_cifar(p=500, resize=None):
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if resize is not None:
        test_list.append(transforms.Resize((resize, resize), interpolation=3))

    transform_test = transforms.Compose(test_list)

    testset = torchvision.datasets.CIFAR10(
        root='/home/lpetrini/data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=2)

    imgs, y = next(iter(testloader))
    return imgs, y


def load_mnist(p=500, fashion=False):
    if not fashion:
        testset = torchvision.datasets.MNIST(
            root='/home/lpetrini/data/mnist', train=False, download=True, transform=transforms.ToTensor())
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = torchvision.datasets.FashionMNIST(
            root='/home/lpetrini/data/fashionmnist', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=2)

    imgs, y = next(iter(testloader))
    return imgs, y


def load_svhn(p=500, resize=None, train=False):
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4523, 0.4524, 0.4689), (0.2190, 0.2261, 0.2279)),
    ]
    if resize is not None:
        test_list.append(transforms.Resize((resize, resize), interpolation=3))

    transform_test = transforms.Compose(test_list)

    testset = torchvision.datasets.SVHN(
        root='/home/lpetrini/data/cifar10', split='train' if train else 'test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=2)

    imgs, y = next(iter(testloader))
    return imgs, y


def load_timagenet(dataset, p=500, resize=None, train=False):
    
    imsize = int(dataset[-2:])
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4824, 0.4495, 0.3981), (0.2768, 0.2691, 0.2827))
    ]

    if imsize != 64:
        test_list.append(transforms.Resize((imsize, imsize), interpolation=3))

    transform_test = transforms.Compose(test_list)

    testset = torchvision.datasets.ImageFolder('/home/lpetrini/data/tiny-imagenet-200/val', transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=2)

    imgs, y = next(iter(testloader))
    return imgs, y

def load_imagenet(p=80):
    im = []
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03000684/*.JPEG"))  # tronconeuses
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n02102040/*.JPEG"))  # chien (une race)
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03425413/*.JPEG"))  # station essence
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n01440764/*.JPEG"))  #
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n02979186/*.JPEG"))  #
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03028079/*.JPEG"))  #
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03445777/*.JPEG"))  #
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03888257/*.JPEG"))  #
    images = []
    y = []
    for i, imm in enumerate(im):
        images = images + imm
        y = y + [i for _ in range(len(imm))]
    P = torch.randperm(len(images))[:p].tolist()
    images = [images[i] for i in P]
    y = [y[i] for i in P]
    y = torch.as_tensor(y)
    imgs = torch.stack([image.square(image.load(i)) for i in images])
    return imgs, y


def diffeo_imgs(imgs, cuts, interp='linear', imagenet=False, Ta=2, Tb=2, nT=30, delta=1.):
    """
    Compute noisy and diffeo versions of imgs.

    :param imgs: image samples to deform
    :param cuts: high-frequency cutoff
    :param interp: interpolation method
    :param imagenet
    :param Ta: T1 = Tmin / Ta
    :param Tb: T2 = Tmax * Tb
    :param nT: number of temperatures
    :param delta: if nT = 1, computes diffeo at delta (used for computing diffeo @ delta=1)
    :return: list of dicts, one per value of cut with deformed and noidy images
    """

    data = {}
    data['cuts'] = []

    n = imgs.shape[-2]
    if interp == 'linear':
        data['imgs'] = imgs
    else:
        if imagenet:
            data['imgs'] = torch.stack([rgb_transpose(deform(rgb_transpose(i), 0, 1, interp='gaussian')) for i in imgs])
        else:
            data['imgs'] = torch.stack([deform(i, 0, 1, interp='gaussian') for i in imgs])

    if interp == 'linear_smooth':
        imgs = data['imgs']
        interp = 'linear'

    for c in cuts:
        T1, T2 = temperature_range(n, c)
        Ts = torch.logspace(math.log10(T1 / Ta), math.log10(T2 * Tb), nT)
        if nT == 1:
            Ts = [typical_temperature(delta, c, n)]
        ds = []
        qs = []
        for T in tqdm(Ts):
            # deform images
            if imagenet:
                d = torch.stack([rgb_transpose(deform(rgb_transpose(i), T, c, interp)) for i in imgs])
            else:
                d = torch.stack([deform(i, T, c, interp) for i in imgs])

            # create gaussian noise with exact same norm
            sigma = (d - data['imgs']).pow(2).sum([1, 2, 3], keepdim=True).sqrt()
            eta = torch.randn(imgs.shape)
            eta = eta / eta.pow(2).sum([1, 2, 3], keepdim=True).sqrt() * sigma
            q = data['imgs'] + eta

            ds.append(d)
            qs.append(q)

        defs = torch.stack(ds)
        nois = torch.stack(qs)

        # smoothing after adding noise
        #         if interp == 'gaussian':
        #             nois = torch.stack([deform(i, 0, 1, interp='gaussian') for i in nois])

        data['cuts'] += [{
            'cut': c,
            'temp': Ts,
            'diffeo': defs,
            'normal': nois,
        }]

    return data


def relative_distance(f, os, ds, qs, deno=True):
    """
    Compute stabilites for a function f

    os: [batch, y, x, rgb]      original images
    ds: [T, batch, y, x, rgb]   images + gaussian noise
    qs: [T, batch, y, x, rgb]   diffeo images
    """
    with torch.no_grad():
        f0 = f(os).detach().reshape(len(os), -1)  # [batch, ...]
        deno = torch.cdist(f0, f0).pow(2).median().item() + 1e-10 if deno else 1
        outd = []
        outq = []
        for d, q in zip(ds, qs):
            fd = f(d).detach().reshape(len(os), -1)  # [batch, ...]
            fq = f(q).detach().reshape(len(os), -1)  # [batch, ...]
            outd += [
                (fd - f0).pow(2).median(0).values.sum().item() / deno
            ]
            outq += [
                (fq - f0).pow(2).median(0).values.sum().item() / deno
            ]
        return torch.tensor(outd), torch.tensor(outq)


def select_net(args):
    num_ch = 1 if 'mnist' in args.dataset else 3
    num_classes = 1 if args.loss == 'hinge' else 10
    imsize = 28 if 'mnist' in args.dataset else 32
    try:
        args.pretrained
    except:
        args.pretrained = 0
    if not args.pretrained:
        if 'VGG' in args.net:
            if 'bn' in args.net:
                bn = True
                net_name = args.net[:-2]
            else:
                net_name = args.net
                bn = False
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
            net = ResNet50(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'LeNet':
            net = LeNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MobileNetV2':
            net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'EfficientNetB0':
            net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'DenseNetL2':
            net = DenseNetL2(num_ch=num_ch * imsize ** 2, num_classes=num_classes)
        if args.net == 'DenseNetL4':
            net = DenseNetL4(num_ch=num_ch * imsize ** 2, num_classes=num_classes)
        if args.net == 'DenseNetL6':
            net = DenseNetL6(num_ch=num_ch * imsize ** 2, num_classes=num_classes)
    else:
        cfg.merge_from_file(f'./models/pretrained/configs/{args.net}.yaml')
        cfg.freeze()
        net = build_EfficientNet(cfg)
    return net


def logmean(x, vec=True):
    if vec:
        x = np.vstack(x)
        return np.exp(np.mean(np.log(x), axis=0))
    else:
        return np.exp(np.mean(np.log(x)))

def std(x, vec=True):
    if vec:
        x = np.vstack(x)
        return np.std(x, axis=0)
    else:
        return np.std(x)
