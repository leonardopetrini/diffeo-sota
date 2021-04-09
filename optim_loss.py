from torch import nn
import torch.optim as optim

def loss_func(args, f, y):
    if args.loss == 'cross_entropy':
        return nn.functional.cross_entropy(args.alpha * f, y, reduction='mean')
    if args.loss == 'hinge':
        return ((1 - args.alpha * f * y).relu()).mean()

def opt_algo(args, net):

    args.lr = args.lr / args.alpha ** args.alphapowerloss

    if args.optim == 'sgd':
        if not args.pretrained:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
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