import torch
from torch.utils.data import Dataset

import itertools


class TwoPointsDataset(Dataset):
    """
        Two points dataset.
    """

    def __init__(self, xi, d=32, ch=1, gap=0, pbc=False, norm='Linf', train=True,
                 transform=None, normalize=True, testsize=5000, background_noise=0, labelling='distance'):
        """

        :param xi: classes separation scale
        :param d: data samples have size `ch x d x d`
        :param ch: number of channels (they are all equal)
        :param gap: gap between `+` or `-` class in terms of the separation scale `xi`
        :param pbc: periodic boundary conditions flag
        :param norm: calculate distance in norm `L2` or `Linf`
        :param train: train / test set flag
        :param transform: transformations to apply to `x`
        :param normalize: normalize data-samples.
        """

        self.d = d
        self.ch = ch
        self.normalize = normalize
        self.background_noise = background_noise

        xp, xm = twopoints_coordinates(xi=xi, d=d, gap=gap, pbc=pbc, norm=norm, labelling=labelling)

        if train:
            xp, xm = xp[: -testsize // 2], xm[: -testsize // 2]
        else:
            xp, xm = xp[-testsize // 2 :], xm[-testsize // 2 :]

        P = min(len(xp), len(xm))
        if train:
            assert P > 0, 'not enough samples for the chosen values of d, xi, gap!!'
            print(f'Max train-set size = {2 * P}')

        self.coordinates = torch.cat([xp[:P], xm[:P]], 0)
        self.targets = torch.zeros(2 * P, dtype=int)
        self.targets[P:] += 1
        self.targets = self.targets.tolist()

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.zeros(self.d, self.d)
        y = self.targets[idx]

        for i, j in self.coordinates[idx]:
            x[i, j] += 1

        if self.normalize:
            m = 2 / self.d ** 2
            # x = (x - m) / (m - m ** 2) ** .5
            x = x / m

        if self.transform:
            x = self.transform(x)

        if self.background_noise:
            g = torch.Generator()
            g.manual_seed(idx)
            x += torch.randn(x.shape, generator=g) * self.background_noise

        return x[None].expand(self.ch, self.d, self.d), y


def twopoints_coordinates(xi, d=32, gap=2, pbc=False, norm='L2', labelling='distance'):
    """
    Returns white-pixels coordinates for the TwoPoints dataset.

    :param xi: classes separation scale
    :param d: data samples have size `1 x d x d`
    :param gap: gap between `+` or `-` class in terms of the separation scale `xi`
    :param pbc: periodic boundary conditions flag
    :param norm: calculate distance in norm `L2` or `Linf`
    :param labelling: labelling function to use, either measure distance between pixel, or 1 if pixels in the same quadrant
    :return (torch.tensor, torch.tensor): (minus_label_coordinates, plus_label_coordinates)
    """

    assert labelling in ['distance', 'quadrant'], "Labelling must be either distance or quadrant based!"

    x = torch.tensor(
        list(itertools.combinations(itertools.permutations(range(d), 2), 2))
    )

    if labelling == 'distance':
        xp, xm = distance_labelling(x, d, xi, gap, pbc, norm)
    else:
        xp, xm = quadrant_labelling(x, d)

    g = torch.Generator() # create new random number generator to avoid conflict with main RNG seed
    g.manual_seed(123456789)
    Pp, Pm = torch.randperm(xp.shape[0], generator=g), torch.randperm(xm.shape[0], generator=g)

    return xp[Pp], xm[Pm]

def distance_labelling(x, d, xi, gap, pbc, norm):
    dist = x.diff(dim=-2).abs()

    if pbc:
        dist = torch.cat([dist, d - dist], dim=-2).min(dim=-2).values
    else:
        dist = dist.squeeze()

    if norm == 'Linf':
        dist = dist.max(dim=-1).values
    elif norm == 'L2':
        dist = dist.pow(2).sum(dim=-1).sqrt()
    else:
        raise ValueError('`norm` must be either `L2` or `Linf`!')

    return x[dist >= xi + gap / 2], x[dist < xi - gap / 2]


def quadrant_labelling(x, d):
    q = x // (d // 2)
    p = (q.diff(dim=-2) == 0).prod(dim=-1).squeeze().bool()
    return x[p], x[~p]

# class TwoPointsDataset(Dataset):
#     """
#         Two points dataset.
#     """
#
#     def __init__(self, xi, d=32, ch=1, gap=0, pbc=False, norm='Linf', train=True, transform=None, normalize=True,
#                  testsize=5000):
#         """
#
#         :param xi: classes separation scale
#         :param d: data samples have size `ch x d x d`
#         :param ch: number of channels (they are all equal)
#         :param gap: gap between `+` or `-` class in terms of the separation scale `xi`
#         :param pbc: periodic boundary conditions flag
#         :param norm: calculate distance in norm `L2` or `Linf`
#         :param train: train / test set flag
#         :param transform: transformations to apply to `x`
#         :param normalize: normalize data-samples.
#         """
#
#         self.d = d
#         self.ch = ch
#         self.normalize = normalize
#
#         #         xp, xm = twopoints_coordinates(xi=xi, d=d, gap=gap, pbc=pbc, norm=norm)
#
#         xp, xm = make_classes(xi, gap, ch, d=d, pbc=pbc, norm=norm)
#
#         if train:
#             xp, xm = xp[: -testsize // 2], xm[: -testsize // 2]
#         else:
#             xp, xm = xp[-testsize // 2:], xm[-testsize // 2:]
#
#         P = min(len(xp), len(xm))
#         if train:
#             assert P > 0, 'not enough samples for the chosen values of d, xi, gap!!'
#             print(f'Max train-set size = {2 * P}')
#
#         self.coordinates = torch.cat([xp[:P], xm[:P]], 0)
#         self.targets = torch.zeros(2 * P, dtype=int)
#         self.targets[P:] += 1
#         self.targets = self.targets.tolist()
#
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.targets)
#
#     def __getitem__(self, idx):
#         """
#         :param idx: sample index
#         :return (torch.tensor, torch.tensor): (sample, label)
#         """
#
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         x = torch.zeros(self.ch, self.d, self.d)
#         y = self.targets[idx]
#
#         for ch in range(self.ch):
#             for i, j in self.coordinates[idx, ch]:
#                 x[ch, i, j] += 1
#
#         if self.normalize:
#             m = 2 / self.d ** 2
#             x = (x - m) / (m - m ** 2) ** .5
#
#         if self.transform:
#             x = self.transform(x)
#
#         if self.ch == 2:
#             x = torch.cat([x, torch.zeros(1, *x[0].shape)])
#
#         return x, y
#
#
# def measure_distance(coordinates, d, norm, pbc):
#     dist = coordinates.diff(dim=-2).abs()
#     if pbc:
#         dist = torch.cat([dist, d - dist], dim=-2).min(dim=-2).values
#     else:
#         dist = dist.squeeze()
#
#     if norm == 'Linf':
#         dist = dist.max(dim=-1).values
#     elif norm == 'L2':
#         dist = dist.pow(2).sum(dim=-1).sqrt()
#     else:
#         raise ValueError('`norm` must be either `L2` or `Linf`!')
#
#     return dist
#
#
# def twopoints_coordinates(xi, gap, d=32, pbc=False, norm='L2'):
#     """
#     Returns white-pixels coordinates for the TwoPoints dataset.
#
#     :param xi: classes separation scale
#     :param d: data samples have size `1 x d x d`
#     :param gap: gap between `+` or `-` class in terms of the separation scale `xi`
#     :param pbc: periodic boundary conditions flag
#     :param norm: calculate distance in norm `L2` or `Linf`
#     :return (torch.tensor, torch.tensor): (minus_label_coordinates, plus_label_coordinates)
#     """
#
#     coordinates = torch.tensor(
#         list(itertools.combinations(itertools.permutations(range(d), 2), 2))
#     )
#
#     dist = measure_distance(coordinates, d, norm, pbc)
#
#     lm = dist >= xi + gap / 2
#     lp = dist < xi - gap / 2
#
#     coordinates = coordinates[lp + lm]
#
#     centers = coordinates.float().mean(dim=-2)
#
#     return coordinates, centers, lp[lp + lm]
#
#
# def make_classes(xi, gap, ch, d=32, pbc=False, norm='L2', N=1000):
#     x, center, label = twopoints_coordinates(xi, gap, d, pbc, norm)
#
#     if ch == 1:
#         xp, xm = x[label, None], x[~label, None]
#
#     elif ch == 2:
#         P = torch.randperm(len(x))
#         x, center, label = x[P], center[P], label[P]
#         x1, center1, label1 = x[:N], center[:N], label[:N]
#         x2, center2, label2 = x[N:2 * N], center[N:2 * N], label[N:2 * N]
#         xx = torch.stack([torch.stack([i, j]) for i in x1 for j in x2])  # [sample, ch, point, coord]
#         cc = torch.stack([torch.stack([i, j]) for i in center1 for j in center2])  # [sample, ch, coord]
#         ll = torch.stack([torch.stack([i, j]) for i in label1 for j in label2])  # [sample, ch]
#
#         dist = measure_distance(cc, d, norm, pbc)
#
#         lm = dist >= xi + gap / 2
#         lp = dist < xi - gap / 2
#
#         xx = xx[lp + lm]
#         label = ll[lp + lm].prod(dim=-1) * lp[lp + lm]
#
#         xp, xm = xx[label], xx[1 - label]
#
#     else:
#         raise
#
#     g = torch.Generator()  # create new random number generator to avoid conflict with main RNG seed
#     g.manual_seed(123456789)
#     Pp, Pm = torch.randperm(xp.shape[0], generator=g), torch.randperm(xm.shape[0], generator=g)
#
#     return xp[Pp], xm[Pm]
