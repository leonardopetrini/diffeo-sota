'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import subprocess

from models import *
import copy
from functools import partial

import sys
import warnings
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
try:
    from transform import Diffeo
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot use diffeo-transform! "
                  "Find the Module in https://github.com/pcsl-epfl/diffeomorphism.")

def loss_func(args, f, y):
    if args.loss == 'cross_entropy':
        return nn.functional.cross_entropy(args.alpha * f, y, reduction='mean')
    if args.loss == 'hinge':
        return ((1 - args.alpha * f * y).relu()).mean()

def opt_algo(args, net):

    args.lr = args.lr / args.alpha ** args.alphapowerloss

    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
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

    trainloader, testloader, net0 = init(args)

    dynamics = [] if args.save_dynamics else None
    loss = []
    best = dict()

    for net, epoch, losstr in train(args, trainloader, net0, criterion):

        loss.append(losstr)

        # Avoid computing accuracy each and every epoch if dataset is small
        if epoch % (args.epochs // 250) != 0: continue

        acc = test(args, testloader, net, net0, criterion)

        if acc > best_acc:
            best['acc'] = acc
            best['epoch'] = epoch
            if args.save_best_net:
                best['net'] = copy.deepcopy(net.state_dict())
            if args.save_dynamics:
                dynamics.append(best)
            best_acc = acc
            print('BEST ACCURACY HERE !!!')
        elif args.save_dynamics and (epoch % 10 == 0):
            dynamics.append({
                'acc': acc,
                'epoch': epoch,
                'net': copy.deepcopy(net.state_dict())
            })
        if losstr == 0:
            break

    out = {
        'args': args,
        'train loss': loss,
        'dynamics': dynamics,
        'best': best,
        # 'net0': copy.deepcopy(net0.state_dict())
    }
    yield out


def train(args, trainloader, net0, criterion):

    net = copy.deepcopy(net0)

    optimizer, scheduler = opt_algo(args, net)
    print(f'Training for {args.epochs} epochs')

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if args.featlazy:
                with torch.no_grad():
                    out0 = net0(inputs)
                loss = criterion(outputs - out0, targets)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if args.loss != 'hinge':
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            else:
                correct += ((outputs - out0) * targets > 0).sum().item()
            total += targets.size(0)

        print(f"[Train epoch {epoch} / {args.epochs}][tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
              f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]")

        scheduler.step()

        yield net, epoch, train_loss/(batch_idx+1)


def test(args, testloader, net, net0, criterion):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)

            if args.featlazy:
                out0 = net0(inputs)
                loss = criterion(outputs - out0, targets)
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            if args.loss != 'hinge':
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            else:
                correct += ((outputs - out0) * targets > 0).double().sum().item()
            total += targets.size(0)

        print(
            f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
            f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]")

    return 100.*correct/total

def init(args):

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
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--save_best_net", type=int, default=0)
    parser.add_argument("--save_dynamics", type=int, default=0)

    parser.add_argument("--featlazy", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--alphapowerloss", type=int, default=1)

    parser.add_argument("--diffeo", type=int, required=True)
    parser.add_argument("--onlydiffeo", type=int, default=0)
    parser.add_argument("--random_crop", type=int, default=0)
    parser.add_argument("--hflip", type=int, default=1)

    parser.add_argument("--sT", type=float, default=2.)
    parser.add_argument("--rT", type=float, default=1.)
    parser.add_argument("--scut", type=float, default=2.)
    parser.add_argument("--rcut", type=float, default=1.)
    parser.add_argument("--cutmax", type=int, default=5)
    parser.add_argument("--cutmin", type=int, default=10)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

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