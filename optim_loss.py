from torch import nn
import torch.optim as optim


def loss_func(args, o, y):
    """
    Compute the loss.
    :param args: parser args
    :param o: network prediction
    :param y: true labels
    :return: value of the loss
    """

    if args.loss == 'cross_entropy':
        loss = nn.functional.cross_entropy(args.alpha * o, y, reduction='mean')
    elif args.loss == 'hinge':
        loss = ((1 - args.alpha * o * y).relu()).mean()
    else:
        raise NameError('Specify a valid loss function in [cross_entropy, hinge]')

    return loss


def measure_accuracy(args, out, out0, targets, correct, total):
    """
        Compute out accuracy on targets. Returns running number of correct and total predictions.
    """
    if args.loss != 'hinge':
        _, predicted = out.max(1)
        correct += predicted.eq(targets).sum().item()
    else:
        correct += ((out - out0) * targets > 0).sum().item()

    total += targets.size(0)

    return correct, total


def opt_algo(args, net):
    """
    Define training optimizer and scheduler.
    :param args: parser args
    :param net: network function
    :return: torch scheduler and optimizer.
    """

    # rescale loss by alpha or alpha**2 if doing feature-lazy
    args.lr = args.lr / args.alpha ** args.alphapowerloss

    if args.optim == 'sgd':
        if not args.pretrained:
            # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) ## 5e-4
            optimizer = optim.SGD([
                {
                    "params": (p for p in net.parameters() if len(p.shape) != 1),
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": (p for p in net.parameters() if len(p.shape) == 1),
                    "weight_decay": 0,
                },
            ], lr=args.lr, momentum=0.9) ## 5e-4
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
        # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay) ## 1e-5
        optimizer = optim.Adam([
                {
                    "params": (p for p in net.parameters() if len(p.shape) != 1),
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": (p for p in net.parameters() if len(p.shape) == 1),
                    "weight_decay": 0,
                },
            ], lr=args.lr) ## 1e-5
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
