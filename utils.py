import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy

import sys
import math
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
sys.path.insert(0, '/home/lpetrini/feature_lazy/')
from arch import CV, FC, FixedAngles, FixedWeights, Wide_ResNet, Conv1d, CrownInit
from dataset import *

from diff import *
from transform import *
import image
from image import *
from etc import *
from tqdm.auto import tqdm
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import functools
from itertools import product
from grid import load
from scipy.stats import beta
from cycler import cycler

from models import *
from models.pretrained import *

from collections import OrderedDict
import pandas as pd

from matplotlib import rc, rcParams
rc('text', usetex=True)
rcParams['font.family'] = 'DejaVu Sans'

from functools import partial

prefix = '/home/lpetrini/results/'


def reset_cycler():
    plt.rc('axes', prop_cycle=(
    cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])))

    
def identity(x):
    return x


def logmean(x, vec=True):
    if vec:
        x = np.vstack(x)
        return np.exp(np.mean(np.log(x), axis=0))
    else:
        return np.exp(np.mean(np.log(x)))

    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    

def triangle(a, b, c, d=None, slope=None, other=False, color=None, fmt="{:.2f}", textpos=None):
    import math
    
    if slope is not None and d is None:
        d = math.exp(math.log(c) + slope * (math.log(b) - math.log(a)))
    if slope is not None and c is None:
        c = math.exp(math.log(d) - slope * (math.log(b) - math.log(a)))
    if color is None:
        color = 'k'

    plt.plot([a, b], [c, d], color=color)
    if other:
        plt.plot([a, b], [c, c], color=color)
        plt.plot([b, b], [c, d], color=color)
    else:
        plt.plot([a, b], [d, d], color=color)
        plt.plot([a, a], [c, d], color=color)

    s = (math.log(d) - math.log(c)) / (math.log(b) - math.log(a))
    if other:
        x = math.exp(0.7 * math.log(b) + 0.3 * math.log(a))
        y = math.exp(0.7 * math.log(c) + 0.3 * math.log(d))
    else:
        x = math.exp(0.7 * math.log(a) + 0.3 * math.log(b))
        y = math.exp(0.7 * math.log(d) + 0.3 * math.log(c))
    if textpos:
        x = textpos[0]
        y = textpos[1]
    plt.annotate(fmt.format(s), (x, y), horizontalalignment='center', verticalalignment='center')
    return s


def typical_temperature(delta, cut):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return delta ** 2 / (0.28 * log + 0.7)


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
                                    transforms.Normalize((0.5,),(0.5,))])
        testset = torchvision.datasets.FashionMNIST(
            root='/home/lpetrini/data/fashionmnist', train=False, download=True, transform=transform)
        
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=2)

    imgs, y = next(iter(testloader))
    return imgs, y

def load_imagenet(p=80):
    im = []
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03000684/*.JPEG")) # tronconeuses
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n02102040/*.JPEG")) # chien (une race)
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03425413/*.JPEG")) # station essence
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n01440764/*.JPEG")) # 
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n02979186/*.JPEG")) #
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03028079/*.JPEG")) # 
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03445777/*.JPEG")) # 
    im.append(glob.glob("/home/mgeiger/datasets/imagenette2-320/val/n03888257/*.JPEG")) # 
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


def diffeo_imgs(imgs, cuts, interp='linear', imagenet=False, Ta=2, Tb=2, nT = 30, delta=1.):

    data = {}
    data['cuts'] = []

    n = imgs.shape[-2]
    print(n)
    
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
            Ts = [typical_temperature(delta, c)]
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


def relative_distance(f, os, ds, qs, avgtype='median', y=None, deno=True):
    """
    os: [batch, y, x, rgb]
    ds: [T, batch, y, x, rgb]
    qs: [T, batch, y, x, rgb]
    """
    with torch.no_grad():
        f0 = f(os).detach().reshape(len(os), -1)  # [batch, ...]
        if avgtype == 'mean':
            deno = torch.cdist(f0, f0).pow(2).mean().item() if deno else 1

            outd = []
            outq = []
            for d, q in zip(ds, qs):        
                fd = f(d).detach().reshape(len(os), -1)  # [batch, ...]
                fq = f(q).detach().reshape(len(os), -1)  # [batch, ...]
                outd += [
                    (fd - f0).pow(2).mean(0).sum().item() / deno
                ]
                outq += [
                    (fq - f0).pow(2).mean(0).sum().item() / deno
                ]
        elif avgtype == 'median':
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
        elif avgtype == 'local':
            outd = []
            outq = []
            for d, q in zip(ds, qs):        
                fd = f(d).detach().reshape(len(os), -1)  # [batch, ...]
                fq = f(q).detach().reshape(len(os), -1)  # [batch, ...]
                outd += [
                    (fd - f0).div(f0).norm(dim=1).median().item()
                ]
                outq += [
                    (fq - f0).div(f0).norm(dim=1).median().item()
                ]
        elif avgtype == 'median-inclass':
            deno = torch.tensor([], device=f0.device)
            for i in range(y.unique().shape[0]):
                deno = torch.cat([deno, torch.cdist(f0[y == i], f0[y == i]).reshape(-1)])
            deno = deno.pow(2).median().item()

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
        else:
            raise
        return torch.tensor(outd), torch.tensor(outq)

    
def probability_change(f, os, ds, qs, y):
    """
    os: [batch, y, x, rgb]
    ds: [T, batch, y, x, rgb]
    qs: [T, batch, y, x, rgb]
    """
    with torch.no_grad():        
        outd = []
        outq = []
        for d, q in zip(ds, qs):        
            pd = f(d).detach().softmax(dim=1)  # [batch, ...]
            pq = f(q).detach().softmax(dim=1)  # [batch, ...]
            outd.append(torch.stack([pd[i, y[i]] for i in range(len(y))]).mean())
            outq.append(torch.stack([pq[i, y[i]] for i in range(len(y))]).mean())
        return torch.tensor(outd).to('cpu'), torch.tensor(outq).to('cpu')
    

# def relative_distance(f, os, ds, qs):
#     """
#     os: [batch, y, x, rgb]
#     ds: [T, batch, y, x, rgb]
#     qs: [T, batch, y, x, rgb]
#     """
#     with torch.no_grad():
#         f0 = f(os).detach().reshape(len(os), -1)  # [batch, ...]
# #         deno = torch.cdist(f0, f0).pow(2).median().item()

#         outd = []
#         outq = []
#         for d, q in zip(ds, qs):        
#             fd = f(d).detach().reshape(len(os), -1)  # [batch, ...]
#             fq = f(q).detach().reshape(len(os), -1)  # [batch, ...]
#             outd += [
#                 (fd - f0).div(f0).norm(dim=1).median().item()
#             ]
#             outq += [
#                 (fq - f0).div(f0).norm(dim=1).median().item()
#             ]
    
#     return torch.tensor(outd), torch.tensor(outq)


def computeR(f, os, ds, qs):
    """
    os: [batch, y, x, rgb]
    ds: [T, batch, y, x, rgb]
    qs: [T, batch, y, x, rgb]
    """
    with torch.no_grad():
        os, ds, qs = os.double(), ds.double(), qs.double()
        f = f.double()
        f0 = f(os).detach().reshape(len(os), -1)
        outr = []
        for d, q in zip(ds, qs):        
            fd = f(d).detach().reshape(len(os), -1)  # [batch, ...]
            fq = f(q).detach().reshape(len(os), -1)  # [batch, ...]        
            outr += [
               ((fd - f0).pow(2) / (fq - f0).pow(2)).log().mean().exp().item()
        ]
        return torch.tensor(outr)

def imshow_cifar(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.gca().imshow(inp)
    
    
def example(fig, T, cut):
    
    a, b = fig.transFigure.inverted().transform(mainax.transData.transform([cut, T]))

    w = 0.08
    
    ax = fig.add_axes([a - w/2, b - w/2, w, w])
    
    i = imgs[image_id]
    i = deform(i, T, int(cut))
    imshow_cifar(i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$E={}$'.format(texnum(typical_energy(n, T, cut), '{:.1f}')), fontsize=6)

    
def fhl(f, x):
    '''Compute first hidden layer output of FC net'''
    W = getattr(f, "W{}".format(0))
    W = torch.cat(list(W))
    h = W.shape[0]
    B = getattr(f, "B0")
    x = x @ (W.t() / h ** 0.5)
    return torch.relu(x + B)

def select_net(args):
    num_ch = 1 if 'mnist' in args.dataset else 3
    num_classes = 1 if args.loss == 'hinge' else 10
    imsize = 28 if 'mnist' in args.dataset else 32
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
        if args.net == 'ConvNetL4':
            net = ConvNetL4(num_ch=num_ch, num_classes=num_classes)
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


def finished(r, ndp=0.01):
    return (r['regular']['dynamics'][-1]['train']['nd'] / r['args'].ptr < ndp)

from arch import FC
def load_data_with_f(filename, predicate=None, t=-1, times='all', device='cpu', load_dataset=True, ndp=.1, dyns=['regular']):
    if predicate == None:
        predicate = lambda a: True

    runs = load('/home/lpetrini/results/' + filename, predicate=predicate)
    runs = [r for r in runs if finished(r, ndp)]
    print(f'Loaded {len(runs)} runs')
    
    for r in runs:
        args = r['args']

        if args.dtype == 'float64':
            torch.set_default_dtype(torch.float64)
        if args.dtype == 'float32':
            torch.set_default_dtype(torch.float32)

        if load_dataset:
            [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
            args.dataset,
            (args.pte, args.ptk, args.ptr),
            (args.seed_testset + args.pte, args.seed_kernelset + args.ptk, args.seed_trainset + args.ptr),
            args.d,
            (args.data_param1, args.data_param2),
            device,
            torch.get_default_dtype()
            )
        else:
            [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = [(None, None, None), (None, None, None), (None, None, None)]
        if args.d is not None:
            dim = args.d
        else:
            if 'cifar' in args.dataset:
                dim = 32 ** 2 * 3
            elif 'mnist' in args.dataset:
                dim = 28 ** 2
            
        torch.manual_seed(0)

        if args.act == 'relu':
            _act = torch.relu
        elif args.act == 'tanh':
            _act = torch.tanh
        elif args.act == 'softplus':
            _act = torch.nn.functional.softplus
        elif args.act == 'swish':
            _act = swish
        else:
            raise ValueError('act not specified')

        def __act(x):
            b = args.act_beta
            return _act(b * x) / b
        factor = __act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item()

        def act(x):
            return __act(x) * factor

        _d = abs(act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item() - 1)
        assert _d < 1e-2, _d
        
        torch.manual_seed(args.seed_init + hash(args.alpha) + args.ptr)

        T = [i + 1 for i, d in enumerate(r['regular']['dynamics'][1:]) if d['state']]
        
        if times == 'all':
            ii = np.arange(len(T))
        else:
            ii = np.linspace(0, len(T) - 1, times, dtype=int)
            ii = list(set(ii))
            ii.sort()

        try:
            d = r['regular']['dynamics'][T[ii[t]]]
        except IndexError:
            print('Index error: dynamics is empty, skipping run!')
            continue
        state = d['state']
        state = OrderedDict([(k[2:], state[k]) for k in state])

        if args.arch == 'fc':
            W = torch.cat([state[s] for s in state if 'W0' in s])
            f0 = FC(dim, args.h, 1, args.L, act, args.bias, False, args.var_bias)
            f = FC(dim, args.h, 1, args.L, act, args.bias, False, args.var_bias)
        if args.arch == 'mnas':
            if 'cifar' in args.dataset:
                ch = 3
            else:
                ch = 1
            f0 = MnasNetLike(ch, args.h, 1, args.cv_L1, args.cv_L2, dim=2)
            f = MnasNetLike(ch, args.h, 1, args.cv_L1, args.cv_L2, dim=2)
        f.load_state_dict(state)
        f.eval()
        f = f.to(device)
        f0 = f0.to(device)
        
        dd = [r[dyn]['dynamics'][-1] if dyn in r.keys() else None for dyn in dyns]
        if len(dd) == 1:
            dd = dd[0]        
        yield [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)], f0, f, args, dd
