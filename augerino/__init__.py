'''
Inspired by 'Learning Invariances in Neural Networks', Benton et al.
https://arxiv.org/pdf/2010.11882.pdf

...in progress...
'''

import torch
import torch.nn as nn

import sys
import warnings
sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')  # path to diffeo library
try:
    import diff
except ModuleNotFoundError:
    warnings.warn("Diffeo Module not found, cannot use diffeo-transform! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")


class Augerino(nn.Module):
    def __init__(self, f, aug, ncopies=4):
        super().__init__()
        self.aug = aug
        self.f = f
        self.n = ncopies

    def forward(self, x):
        if self.training:
            return self.f(self.aug(x))
        else:
            return torch.stack([self.f(self.aug(x)) for _ in range(self.n)]).mean(dim=0)


class DiffeoAug(nn.Module):
    def __init__(self, logsT, logrT, logscut, logrcut, cutmin, cutmax):
        super().__init__()

        self.logsT = nn.Parameter(torch.tensor(logsT))
        self.logrT = nn.Parameter(torch.tensor(logrT))
        self.logscut = nn.Parameter(torch.tensor(logscut))
        self.logrcut = nn.Parameter(torch.tensor(logrcut))
        self.cutmin = cutmin
        self.cutmax = cutmax

    def forward(self, imgs):
        """
        Args:
            img (Tensor): Image(s) to be 'diffeomorphed'.
        Returns:
            Tensor: Diffeo image(s).
        """

        # image size
        n = imgs.shape[-1]
        ten = torch.tensor(10.)

        sT = ten.pow(self.logsT)
        scut = ten.pow(self.logscut)
        rT = ten.pow(self.logrT)
        rcut = ten.pow(self.logrcut)

        betaT = torch.distributions.beta.Beta(sT - sT / (rT + 1), sT / (rT + 1), validate_args=None)
        betacut = torch.distributions.beta.Beta(scut - scut / (rcut + 1), scut / (rcut + 1), validate_args=None)

        cut = (betacut.sample() * (self.cutmax + 1 - self.cutmin) + self.cutmin).int().item()
        T1, T2 = diff.temperature_range(n, cut)
        T = (betaT.sample() * (T2 - T1) + T1)

        # applying the same diffeo to all images in a batch, maybe not the best thing to do but fast.
        return diff.deform(imgs, T, cut)
        # return torch.stack([diff.deform(img, T, cut) for img in imgs])

    def __repr__(self):
        return self.__class__.__name__ + f'LogParams: (sT={self.logsT}, rT={self.logrT}, scut={self.logscut}, rcut={self.logrcut})'

