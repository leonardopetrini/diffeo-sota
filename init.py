import torch

from datasets import dataset_initialization
from models import model_initialization

import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def init_fun(args):
    """
        Initialize dataset and architecture.
    """
    torch.manual_seed(args.seed_init)

    trainloader, testloader, imsize, num_classes = dataset_initialization(args)

    net = model_initialization(args, image_size=imsize, num_classes=num_classes)

    return trainloader, testloader, net



