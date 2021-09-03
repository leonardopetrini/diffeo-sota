import torch
from torch import nn
import torch.optim as optim

import sys
import warnings
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')  # path to diffeo library
try:
    from diff import deform
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot use diffeo-transform! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")


def loss_func(args, o, y, x=None, f=None):

    if args.loss == 'cross_entropy':
        loss = nn.functional.cross_entropy(args.alpha * o, y, reduction='mean')
    if args.loss == 'hinge':
        loss = ((1 - args.alpha * o * y).relu()).mean()

    if args.diffeo_decay and x is not None:
        # eps = 1e-10
        l = args.diffeo_decay
        d, g = diffeo(x, args)
        od = f(d)
        # og = f(g)
        # D = (od - o).pow(2).mean()
        # G = (og - o).pow(2).mean()
        # reg = (D / (G + eps)).clamp(1e-8, 1e2)
        # return loss + l * reg.sqrt()
        reg = nn.functional.kl_div(od.softmax(dim=1), o.softmax(dim=1), reduction='batchmean', log_target=False)
        reg = 1 if torch.isnan(reg) else reg
        return loss + l * reg
    else:
        return loss

def opt_algo(args, net):

    # rescale loss by alpha or alpha**2 if doing feature-lazy
    args.lr = args.lr / args.alpha ** args.alphapowerloss

    if args.optim == 'sgd':
        if not args.pretrained:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            # for transfer learning, learning rate of last -newly initialized- layer is 10x lr of the rest
            my_list = ['_fc.weight', '_fc.bias']
            classifier_params = list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))
            base_params = list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))
            classifier_params = [p[1] for p in classifier_params]
            base_params = [p[1] for p in base_params]
            lr = 0.01
            optimizer = optim.SGD([
                {
                    "params": base_params,
                    "lr": lr * 0.1,
                },
                {
                    "params": classifier_params,
                    "lr": lr
                }],
                momentum=0.9, weight_decay=0.001, nesterov=True)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NameError('Specify a valid optimizer [Adam, (S)GD]')
    if args.scheduler == 'cosineannealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 0.8)
    elif args.scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=1.)
    elif args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    else:
        raise NameError('Specify a valid scheduler [cosineannealing, exponential, none]')

    return optimizer, scheduler

def diffeo(xs, args):
    assert args.dataset == 'cifar10', 'Diffeo decay only implemented for CIFAR10!'
    c = 3
    interp = 'linear'
    T = 0.001131789
    norm = 338.3901

    d = torch.stack([deform(x, T, c, interp) for x in xs])

    eta = torch.randn(d.shape, device=args.device)
    eta = eta / eta.pow(2).sum([1, 2, 3], keepdim=True).sqrt() * norm

    return d, xs + eta