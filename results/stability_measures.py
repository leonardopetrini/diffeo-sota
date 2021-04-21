"""
Experiments for the paper ""
run this file to compute stability measures on results.
see below (*) for how to get results by running the main file.
[...polishing in progress...]


Dependencies (other than common ones):
- Experiments are run using `grid` https://github.com/mariogeiger/grid/tree/master/grid
- `diffeomorphism` https://github.com/pcsl-epfl/diffeomorphism


### TRAININGS ### (*)

 - conv. nets at different train set sizes P (arg: ptr)

python -m grid /home/lpetrini/results/diffeopaper --n 16 "
srun --nice --partition gpu --qos gpu --gres gpu:1 --time 24:00:00 --mem 10G
 python main.py --epochs 250 --onlydiffeo 0 --save_best_net 1 --save_dynamics 0 --diffeo 0 --random_crop 1 --hflip 1
 " --batch_size 128 --seed_init 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --net:str 'LeNet' 'ResNet18' 'EfficientNetB0' 
 --dataset:str 'mnist' 'fashionmnist' 'cifar10' --ptr 512 1024 2048 4096 8192 16384 32768 50000
 
 
 - fully connected net at different train set sizes P
 
python -m grid /home/lpetrini/results/diffeopaper --n 12 "
srun --nice --partition gpu --qos gpu --gres gpu:1 --time 24:00:00 --mem 6G
 python main.py --epochs 250 --onlydiffeo 0 --save_best_net 1 --save_dynamics 0 --diffeo 0 --random_crop 1 --hflip 1
 " --batch_size 128 --net:str 'DenseNetL4' --optim:str 'adam' --scheduler:str 'none' 
 --dataset:str 'mnist' 'fashionmnist' 'cifar10' --seed_init 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --ptr 512 1024 2048 4096 8192 16384 32768 50000


 - SOTA nets for test error vs. R_f plot

python -m grid /home/lpetrini/results/diffeopaper --n 32 "
srun --nice --partition gpu --qos gpu --gres gpu:1 --time 24:00:00 --mem 10G
 python main.py --epochs 350 --onlydiffeo 0 --save_best_net 1 --save_dynamics 0 --diffeo 0 --random_crop 1 --hflip 1
 " --batch_size 128 --seed_init 0 1 2 3 4 --net:str 'VGG11' 'VGG16' 'VGG19' 'ResNet34' 'ResNet50' 'ResNet101' 'VGG11bn' 'VGG16bn' 'VGG19bn' 'AlexNet'
 --dataset:str 'cifar10' --ptr 50000


 - EfficientNet(B0-B2), transfer learning from ImageNet
 
python -m grid /home/lpetrini/results/effnet_cifar_pretrained --n 32 "
srun --nice --partition gpu --qos gpu --gres gpu:1 --time 24:00:00 --mem 32G
 python main.py --epochs 100 --onlydiffeo 0 --save_best_net 1 --save_dynamics 0 --diffeo 0 --random_crop 1 --hflip 1 --device cuda --scheduler exponential
 " --batch_size 128 --net:str 'efficientnetB0' 'efficientnetB2' --seed_init 0 1 2 3 4 --pretrained 1 --dataset:str 'cifar10' --ptr 50000

"""

from utils import *

import pandas as pd
import torch

from grid import load

from itertools import product
from functools import partial
from collections import OrderedDict


# results dir
prefix = '/home/lpetrini/results/'
filename = 'diffeopaper'

# using pandas dataframe to store results
df_filename = 'df_diffeopaper'

num_probing_points = 500
cuts = [3, 5, 15]
Ta, Tb = 15, 2
avg = 'median'
device = 'cpu'

nets = ['DenseNetL4', 'LeNet', 'ResNet18', 'EfficientNetB0', 'VGG11', 'DenseNetL2', 'DenseNetL6',
        'VGG16', 'VGG19', 'ResNet34', 'ResNet50', 'VGG11bn', 'VGG16bn', 'VGG19bn', 'AlexNet']
datasets = ['mnist', 'fashionmnist', 'cifar10']

try:
    df = torch.load(f'./dataframes/{df_filename}.torch')
except FileNotFoundError:
    df = pd.DataFrame()

for interp, net in product(['linear', 'gaussian'], nets):
    for dataset in datasets:

        # Load results
        predicate = lambda a: a.dataset == dataset and a.net == net
        runs = load(prefix + filename, predicate=predicate)
        print(f'Loaded {len(runs)} runs')

        # Load dataset
        if 'mnist' in dataset:
            ch = 1
            imgs, y = load_mnist(p=num_probing_points, fashion='fashion' in dataset)
        else:
            ch = 3
            imgs, y = load_cifar(p=num_probing_points)

        # generate max-entropy diffeo and noisy samples
        imgs = imgs.to('cpu')
        data = diffeo_imgs(imgs, cuts=cuts, interp=interp, Ta=Ta, Tb=Tb)
        imgs = data['imgs'].to(device)

        for r in tqdm(runs):
            # if r['best']['acc'] < 20: continue
            args = r['args']
            state = OrderedDict([(k[7:], r['best']['net'][k]) for k in r['best']['net']])

            # initialize network
            f0 = select_net(args)

            # use batch statistics for batch-norm as at init there is no running statistics available
            for c in f0.modules():
                if 'batchnorm' in str(type(c)):
                    c.track_running_stats = False
                    c.running_mean = None
                    c.running_var = None
            f0.eval()
            f0 = f0.to(device)

            if args.ptr == 5e4:     # compute init. quantities only for one P, no need to do it at each P
                for ix, x in enumerate(data['cuts']):

                    # compute stabilities D_f and G_f at init
                    ds, gs = relative_distance(f0, imgs.float(),
                                               x['diffeo'].to(device).float(),
                                               x['normal'].to(device).float(),
                                               )
                    df = df.append({
                        'dataset': dataset,
                        'net': args.net,
                        'seed_init': args.seed_init,
                        'ptr': args.ptr,
                        'cut': cuts[ix],
                        'Ts': (Ta, Tb),
                        'layer': 'f',
                        'trained': False,
                        'scale_batch_size': args.scale_batch_size,
                        'acc': r['best']['acc'],
                        'epoch': r['best']['epoch'],
                        'interp': interp,
                        'avg-type': avg,
                        'D': ds.to('cpu').numpy(),
                        'G': gs.to('cpu').numpy(),
                    }, ignore_index=True)

            ### TRAINED ###

            # load trained state
            f = select_net(args)
            f.load_state_dict(state)

            f.eval()
            f.to(device)

            for ix, x in enumerate(data['cuts']):

                # compute stabilities D_f and G_f for trained net
                ds, gs = relative_distance(f, imgs.float(),
                                           x['diffeo'].to(device).float(),
                                           x['normal'].to(device).float(),
                                           )

                df = df.append({
                    'dataset': dataset,
                    'net': args.net,
                    'seed_init': args.seed_init,
                    'ptr': args.ptr,
                    'cut': cuts[ix],
                    'Ts': (Ta, Tb),
                    'layer': 'f',
                    'trained': True,
                    'scale_batch_size': args.scale_batch_size,
                    'acc': r['best']['acc'],
                    'epoch': r['best']['epoch'],
                    'interp': interp,
                    'avg-type': avg,
                    'D': ds.to('cpu').numpy(),
                    'G': gs.to('cpu').numpy(),
                }, ignore_index=True)

torch.save(df, f'./dataframes/{df_filename}.torch')

# compute relative diffeo stability
df['R'] = df['D'] / df['G']

# compute average results
groups = ['dataset', 'net', 'cut', 'trained', 'interp', 'ptr']

D = df.groupby(groups)['D'].apply(logmean)
G = df.groupby(groups)['G'].apply(logmean)
R = df.groupby(groups)['R'].apply(logmean)
Dstd = df.groupby(groups)['D'].apply(std)
Gstd = df.groupby(groups)['G'].apply(std)
Rstd = df.groupby(groups)['R'].apply(std)

acc = df.groupby(groups)['acc'].apply(partial(logmean, vec=False))
epoch = df.groupby(groups)['epoch'].apply(partial(logmean, vec=False))
