import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy

import sys
import math

sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
from diff import *
from transform import *
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
from cycler import cycler

from models import *
from collections import OrderedDict
import pandas as pd

from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['font.family'] = 'DejaVu Sans'

from functools import partial

prefix = '/home/lpetrini/results/'


def reset_cycler():
    plt.rc('axes', prop_cycle=(
        cycler('color',
               ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                '#17becf'])))


def identity(x):
    return x


def logmean(x, vec=True):
    if vec:
        x = np.vstack(x)
        return np.exp(np.mean(np.log(x), axis=0))
    else:
        return np.exp(np.mean(np.log(x)))


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
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


def load_cifar(p=500):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='/home/lpetrini/data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=2)

    imgs, _ = next(iter(testloader))
    return imgs


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

    imgs, _ = next(iter(testloader))
    return imgs


def diffeo_imgs(imgs, cuts, Ts, interp='linear'):
    data = {}
    data['cuts'] = []

    n = imgs.shape[-1]

    if interp == 'linear':
        data['imgs'] = imgs
    else:
        data['imgs'] = torch.stack([deform(i, 0, 1, interp='gaussian') for i in imgs])

    if interp == 'linear_smooth':
        imgs = data['imgs']
        interp = 'linear'

    for c in cuts:
        ds = []
        qs = []

        for T in tqdm(Ts):
            # deform images
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


def relative_distance(f, os, ds, qs):
    """
    os: [batch, y, x, rgb]
    ds: [T, batch, y, x, rgb]
    qs: [T, batch, y, x, rgb]
    """
    with torch.no_grad():
        f0 = f(os).detach().reshape(len(os), -1)  # [batch, ...]
        deno = torch.cdist(f0, f0).pow(2).mean().item()

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

    return torch.tensor(outd), torch.tensor(outq)


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
    ax = fig.add_axes([a - w / 2, b - w / 2, w, w])
    i = imgs[image_id]
    i = deform(i, T, int(cut))
    imshow_cifar(i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('$E={}$'.format(texnum(typical_energy(n, T, cut), '{:.1f}')), fontsize=6)


def fhl(f, x):
    """Compute first hidden layer output of FC net"""
    W = getattr(f, "W{}".format(0))
    W = torch.cat(list(W))
    h = W.shape[0]
    B = getattr(f, "B0")
    x = x @ (W.t() / h ** 0.5)
    return torch.relu(x + B)
