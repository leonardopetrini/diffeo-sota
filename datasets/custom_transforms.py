import torch


def filter_(x, window_size=10, high_pass=True):
    N = x.shape[1]

    fi = torch.fft.rfft(x, signal_ndim=2, onesided=False, normalized=False)

    if not high_pass:
        fi = torch.roll(fi, shifts=[fi.shape[1] // 2, fi.shape[1] // 2], dims=[1, 2])

    k = torch.arange(-N // 2, N // 2)
    xx, yy = torch.meshgrid(k, k)
    ks = (xx ** 2 + yy ** 2) ** .5
    fi[:, ks > window_size] = 0

    if not high_pass:
        fi = torch.roll(fi, shifts=[fi.shape[1] // 2, fi.shape[1] // 2], dims=[1, 2])

    return torch.fft.irfft(fi, signal_ndim=2, onesided=False, normalized=False)


class LowHighPassFilter(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.B = torch.distributions.bernoulli.Bernoulli(p)

    def forward(self, img):

        # half image size
        n = img.shape[-1] // 4

        if self.B.sample():
            return filter_(img, window_size=torch.randint(15, (1,))[0].item() + 10, high_pass=self.B.sample())
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + f'(Random low or high pass filter with proba p = {self.p})'

class AvgChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be averaged.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return tensor.mean(dim=-3, keepdim=True)

    def __repr__(self):
        return self.__class__.__name__


class GaussianNoiseCorruption(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()).div(torch.tensor(tensor.shape[-3:]).prod().sqrt())
        return tensor + noise * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomSubspaceCorruption(torch.nn.Module):
    def __init__(self, deff, d, std=1.):
        super().__init__()
        # coefficients mapping effective to real space
        self.c = torch.randn(deff, d)
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.reshape(-1, *tensor.shape[-3:])

        p = tensor.shape[0]
        deff = self.c.shape[0]

        # sample noise in effective space
        noise_tilde = torch.randn(p, deff)

        # map noise to real space
        noise = torch.einsum('pn,nd->pd', noise_tilde, self.c).reshape(tensor.shape)

        noise = noise.div(noise.pow(2).sum([1, 2, 3], keepdim=True).sqrt())
        return (tensor + noise * self.std).squeeze(dim=0)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
