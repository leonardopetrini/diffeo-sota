'''
Inspired by 'Learning Invariances in Neural Networks', Benton et al.
https://arxiv.org/pdf/2010.11882.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def expm(A,rtol=1e-4):
    """ assumes A has shape (bs,d,d)
        returns exp(A) with shape (bs,d,d) """
    I = torch.eye(A.shape[-1],device=A.device,dtype=A.dtype)[None].repeat(A.shape[0],1,1)
    return odeint(lambda t, x: A@x,I,torch.tensor([0.,1.]).to(A.device,A.dtype), rtol=rtol)[-1]


class UniformAug(nn.Module):
    """docstring for MLPAug"""

    def __init__(self, gen_scale=10., trans_scale=0.1,
                 epsilon=1e-3):
        super(UniformAug, self).__init__()

        self.trans_scale = trans_scale

        self.width = nn.Parameter(torch.zeros(6))
        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None

    def set_width(self, vals):
        self.width.data = vals

    def transform(self, x):
        bs, _, w, h = x.size()
        weights = torch.rand(bs, 6)
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width)
        weights = weights * width - width.div(2.)

        generators = self.generate(weights)

        ## exponential map

        affine_matrices = expm(generators.cpu()).to(weights.device)

        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        return x_out

    def generate(self, weights):
        """
        return the sum of the scaled generator matrices
        """
        bs = weights.shape[0]

        if self.g0 is None or self.std_batch_size != bs:
            self.std_batch_size = bs

            ## tx
            self.g0 = torch.zeros(3, 3, device=weights.device)
            self.g0[0, 2] = 1. * self.trans_scale
            self.g0 = self.g0.unsqueeze(-1).expand(3, 3, bs)

            ## ty
            self.g1 = torch.zeros(3, 3, device=weights.device)
            self.g1[1, 2] = 1. * self.trans_scale
            self.g1 = self.g1.unsqueeze(-1).expand(3, 3, bs)

            self.g2 = torch.zeros(3, 3, device=weights.device)
            self.g2[0, 1] = -1.
            self.g2[1, 0] = 1.
            self.g2 = self.g2.unsqueeze(-1).expand(3, 3, bs)

            self.g3 = torch.zeros(3, 3, device=weights.device)
            self.g3[0, 0] = 1.
            self.g3[1, 1] = 1.
            self.g3 = self.g3.unsqueeze(-1).expand(3, 3, bs)

            self.g4 = torch.zeros(3, 3, device=weights.device)
            self.g4[0, 0] = 1.
            self.g4[1, 1] = -1.
            self.g4 = self.g4.unsqueeze(-1).expand(3, 3, bs)

            self.g5 = torch.zeros(3, 3, device=weights.device)
            self.g5[0, 1] = 1.
            self.g5[1, 0] = 1.
            self.g5 = self.g5.unsqueeze(-1).expand(3, 3, bs)

        out_mat = weights[:, 0] * self.g0
        out_mat += weights[:, 1] * self.g1
        out_mat += weights[:, 2] * self.g2
        out_mat += weights[:, 3] * self.g3
        out_mat += weights[:, 4] * self.g4
        out_mat += weights[:, 5] * self.g5

        # transposes just to get everything right
        return out_mat.transpose(0, 2).transpose(2, 1)

    def forward(self, x):
        return self.transform(x)