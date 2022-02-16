'''
Train SOTA nets on MNIST, FashionMNIST or CIFAR10 with PyTorch.
[see readme.md]
'''
import os
import argparse
import time

from models import *
import copy
from functools import partial

from init import init_fun
from optim_loss import loss_func, opt_algo, measure_accuracy

def run(args):

    best_acc = 0  # best test accuracy
    criterion = partial(loss_func, args)

    trainloader, testloader, net0 = init_fun(args)

    # scale batch size when smaller than train-set size
    if (args.batch_size <= args.ptr) and args.scale_batch_size:
        args.batch_size = args.ptr // 2

    if args.save_dynamics:
        dynamics = [{
            'acc': 0.,
            'epoch': -1,
            'net': copy.deepcopy(net0.state_dict())
        }]
    else:
        dynamics = None

    loss = []
    terr = []
    best = dict()

    for net, epoch, losstr in train(args, trainloader, net0, criterion):

        assert str(losstr) != 'nan', 'Loss is nan value!!'
        loss.append(losstr)

        # avoid computing accuracy each and every epoch if dataset is small and epochs are rescaled
        if epoch > 250:
            if epoch % (args.epochs // 250) != 0: continue

        acc = test(args, testloader, net, criterion, net0)
        terr.append(100 - acc)

        if args.save_dynamics and (epoch in (10 ** torch.linspace(-1, math.log10(args.epochs), 30)).int().unique()):
            # save dynamics at 30 log-spaced points in time
            dynamics.append({
                'acc': acc,
                'epoch': epoch,
                'net': copy.deepcopy(net.state_dict())
            })
        if acc > best_acc:
            best['acc'] = acc
            best['epoch'] = epoch
            if args.save_best_net:
                best['net'] = copy.deepcopy(net.state_dict())
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f'BEST ACCURACY ({acc:.02f}) at epoch {epoch+1} !!')
            out = {
                'args': args,
                'train loss': loss,
                'dynamics': dynamics,
                'best': best,
            }
            yield out
        if losstr == 0:
            break

    out = {
        'args': args,
        'train loss': loss,
        'terr': terr,
        'dynamics': dynamics,
        'best': best,
        'last': copy.deepcopy(net.state_dict()) if args.random_labels or args.save_last_net else None,
    }
    yield out


def train(args, trainloader, net0, criterion):

    net = copy.deepcopy(net0)

    optimizer, scheduler = opt_algo(args, net)
    print(f'Training for {args.epochs} epochs...')

    start_time = time.time()

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs).squeeze()
            if args.featlazy:
                with torch.no_grad():
                    out0 = net0(inputs).squeeze()
                loss = criterion(outputs - out0, targets)
            else:
                out0 = 0
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, out0, targets, correct, total)

        avg_epoch_time = (time.time() - start_time) / (epoch + 1)
        print(f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
              f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
              f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]")

        scheduler.step()

        yield net, epoch, train_loss / (batch_idx + 1)


def test(args, testloader, net, criterion, net0=None):

    net.eval()
    if net0 is None:
        net0 = lambda x: 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs).squeeze()

            if args.featlazy:
                out0 = net0(inputs).squeeze()
                loss = criterion(outputs - out0, targets)
            else:
                out0 = 0
                loss = criterion(outputs, targets)

            test_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, out0, targets, correct, total)

        print(
            f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
            f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]")

    return 100. * correct / total


# timing function
def print_time(elapsed_time):
    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f'{h}h')
    if not (h == 0 and m == 0):
        elapsed_time.append(f'{m:02}m')
    elapsed_time.append(f'{s:02}s')

    return ''.join(elapsed_time)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)
    parser.add_argument("--random_labels", type=int, default=0)
    parser.add_argument("--black_and_white", type=int, default=0)
    parser.add_argument("--group_tiny_classes", type=int, default=0)
    parser.add_argument("--gaussian_corruption_std", type=float, default=0.)
    parser.add_argument("--corruption_subspace_dimension", type=int, default=0)

    parser.add_argument("--xi", type=float, default=5)
    parser.add_argument("--gap", type=float, default=0)
    parser.add_argument("--norm", type=str, default='Linf')
    parser.add_argument("--pbc", type=int, default=0)

    parser.add_argument("--pte", type=int, default=0)
    parser.add_argument("--T", type=float, default=1e-5)
    parser.add_argument("--t", type=float, default=1e-3)


    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_net", type=int, default=-1)
    parser.add_argument("--seed_data", type=int, default=-1)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--pretrained", type=int, default=0)
    parser.add_argument("--random_features", type=int, default=0)

    parser.add_argument("--loss", type=str, default='cross_entropy')
    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--scheduler", type=str, default='cosineannealing')
    parser.add_argument("--param_list", type=int, default=0, help='Make parameters list for NTK calculation')

    # params for simple FCNs and CNNs
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width_factor", type=float, default=1.)
    parser.add_argument("--filter_size", type=int, default=5)
    parser.add_argument("--pooling_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)


    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--rescale_epochs", type=int, default=0)
    parser.add_argument("--save_best_net", type=int, default=0)
    parser.add_argument("--save_last_net", type=int, default=0)
    parser.add_argument("--save_dynamics", type=int, default=0)

    parser.add_argument("--featlazy", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--alphapowerloss", type=int, default=1)

    parser.add_argument("--diffeo", type=int, required=True)
    parser.add_argument("--random_crop", type=int, default=0)
    parser.add_argument("--hflip", type=int, default=1)

    parser.add_argument("--sT", type=float, default=2.)
    parser.add_argument("--rT", type=float, default=1.)
    parser.add_argument("--scut", type=float, default=2.)
    parser.add_argument("--rcut", type=float, default=1.)
    parser.add_argument("--cutmin", type=int, default=1)
    parser.add_argument("--cutmax", type=int, default=15)

    parser.add_argument("--scattering_mode", type=int, default=0)
    parser.add_argument("--J", type=int, default=2)
    parser.add_argument("--L", type=int, default=8)

    parser.add_argument("--diffeo_decay", type=float, default=0.)

    parser.add_argument("--train_filtered", type=int, default=0.)
    parser.add_argument("--filter_p", type=float, default=0.5)


    parser.add_argument("--pickle", type=str, required=False, default='None')
    parser.add_argument("--output", type=str, required=False, default='None')
    args = parser.parse_args()

    if args.pickle == 'None':
        assert args.output != 'None', 'either `pickle` or `output` must be given to the parser!!'
        args.pickle = args.output

    torch.save(args, args.pickle)
    saved = False
    try:
        for res in run(args):
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
