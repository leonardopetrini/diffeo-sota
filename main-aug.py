'''
Train SOTA nets on MNIST, FashionMNIST or CIFAR10 with PyTorch.
[see readme.md]
'''
import torch.optim as optim

import os
import argparse
import subprocess

from models import *
from augerino import *
import copy
from functools import partial

from init import init_fun


def loss_func(args, o, y, model):
    shutdown = model.aug.logrT < 0
    if args.loss == 'cross_entropy':
        return nn.functional.cross_entropy(o, y, reduction='mean') - args.aug_lambda * (model.aug.logrT + model.aug.logrcut) * shutdown

def opt_algo(args, model):

    params_list = [
        {'name': 'f',
         'params': model.f.parameters(),
         "weight_decay": args.weight_decay},
        {'name': 'aug',
         'params': model.aug.parameters(),
         "weight_decay": 0.}
    ]

    if args.optim == 'sgd':
        optimizer = optim.SGD(params_list, lr=args.lr, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params_list, lr=args.lr)
    else:
        raise NameError('Specify a valid optimizer [Adam, (S)GD]')
    if args.scheduler == 'cosineannealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=1.)
    return optimizer, scheduler


def run(args):

    best_acc = 0  # best test accuracy
    criterion = partial(loss_func, args)

    trainloader, testloader, net = init_fun(args)

    aug = DiffeoAug(logsT=args.sT, logrT=args.rT, logscut=args.scut, logrcut=args.rcut, cutmin=args.cutmin, cutmax=args.cutmax)
    model = Augerino(net, aug, args.ncopies)
    model = model.to(args.device)

    dynamics = [] if args.save_dynamics else None
    loss = []
    best = dict()

    for model, epoch, losstr in train(args, trainloader, model, criterion):

        loss.append(losstr)

        # Avoid computing accuracy each and every epoch if dataset is small
        if epoch % (args.epochs // 250) != 0: continue

        acc = test(args, testloader, model, criterion)
        print(model.aug)

        if acc > best_acc:
            best['acc'] = acc
            best['epoch'] = epoch
            if args.save_best_net:
                best['net'] = copy.deepcopy(model.f.state_dict())
                best['aug'] = copy.deepcopy(model.aug.state_dict())
            if args.save_dynamics:
                dynamics.append(best)
            best_acc = acc
            print('BEST ACCURACY HERE !!!')
        elif args.save_dynamics and (epoch % 10 == 0):
            dynamics.append({
                'acc': acc,
                'epoch': epoch,
                'net': copy.deepcopy(model.f.state_dict()),
                'aug': copy.deepcopy(model.aug.state_dict())
            })
        if losstr == 0:
            break

    out = {
        'args': args,
        'train loss': loss,
        'dynamics': dynamics,
        'best': best,
    }
    yield out


def train(args, trainloader, model, criterion):

    optimizer, scheduler = opt_algo(args, model)
    print(f'Training for {args.epochs} epochs')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, model)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        print(f"[Train epoch {epoch} / {args.epochs}][tr.Loss: {train_loss / (batch_idx + 1):.03f}]"
              f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]")

        scheduler.step()

        yield model, epoch, train_loss / (batch_idx+1)


def test(args, testloader, model, criterion):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, model)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        print(
            f"[TEST][te.Loss: {test_loss / (batch_idx + 1):.03f}]"
            f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]")

    return 100.*correct/total


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--loss", type=str, default='cross_entropy')
    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--scheduler", type=str, default='cosineannealing')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--save_best_net", type=int, default=0)
    parser.add_argument("--save_dynamics", type=int, default=0)

    parser.add_argument("--augerino", default='diffeo', type=str, help='augerino type')
    parser.add_argument('--ncopies', default=4, type=int)
    parser.add_argument('--aug_lambda', default=0.01, type=float)

    parser.add_argument("--sT", type=float, default=1.)
    parser.add_argument("--rT", type=float, default=-1.)
    parser.add_argument("--scut", type=float, default=1.)
    parser.add_argument("--rcut", type=float, default=-1.)
    parser.add_argument("--cutmin", type=int, default=1)
    parser.add_argument("--cutmax", type=int, default=15)

    parser.add_argument("--random_crop", type=int, default=0)
    parser.add_argument("--hflip", type=int, default=1)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    args.diffeo = 0
    args.onlydiffeo = 0

    torch.save(args, args.pickle)
    saved = False
    try:
        for res in run(args):
            res['git'] = git
            with open(args.pickle, 'wb') as f:
                torch.save(args, f, _use_new_zipfile_serialization=False)
                torch.save(res, f, _use_new_zipfile_serialization=False)
                saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise

if __name__ == "__main__":
    main()