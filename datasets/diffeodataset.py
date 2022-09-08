import warnings
from torch.utils.data import TensorDataset

import torchvision
from torchvision import transforms

import sys
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')

try:
    from diff import *
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot deform images! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")


def load_cifar_samples(p=500, device='cpu'):
    """
    :param p: number of samples to load
    :param device: device on which samples are loaded
    :return: x, y: p images and labels from CIFAR10
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    set = torchvision.datasets.CIFAR10(
        root='/home/lpetrini/data/cifar10', train=True, download=True, transform=transform_test)
    loader = torch.utils.data.DataLoader(
        set, batch_size=p, shuffle=False, num_workers=2)

    imgs, y = next(iter(loader))

    return imgs.to(device), y.to(device)


def diffeo_dataset(original, p, cut, T=1e-3, t=1e-5):
    '''
    Build a dataset starting from a single image and using diffeo.
    Classes are separated by a single diffeo of temperature T.
    Inside a class, images are produced with temp. t diffeos.

    :param original: original image to build the dataset on
    :param p: number of samples
    :param cut: cut-off value for the diffeos
    :param T: inter-class temperature
    :param t: intra-class temperature
    :return: torch dataset
    '''

    x1 = deform(original, cut=cut, T=T)[None].repeat(p // 2, *[1 for _ in range(len(original.shape))])
    x2 = deform(original, cut=cut, T=T)[None].repeat(p // 2, *[1 for _ in range(len(original.shape))])

    x = torch.cat([x1, x2])

    inputs = torch.stack([deform(xi, cut=cut, T=t) for xi in x])

    targets = torch.zeros(p, dtype=torch.long)
    targets[p // 2:] += 1
    dataset = TensorDataset(inputs, targets)

    return dataset