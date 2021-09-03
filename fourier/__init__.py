import torch

def Filter(x, window_size=10, high_pass=True):
    N = x.shape[1]

    fi = torch.rfft(x, signal_ndim=2, onesided=False, normalized=False)

    if not high_pass:
        fi = torch.roll(fi, shifts=[fi.shape[1] // 2, fi.shape[1] // 2], dims=[1, 2])

    k = torch.arange(-N // 2, N // 2)
    xx, yy = torch.meshgrid(k, k)
    ks = (xx ** 2 + yy ** 2) ** .5
    fi[:, ks > window_size] = 0

    if not high_pass:
        fi = torch.roll(fi, shifts=[fi.shape[1] // 2, fi.shape[1] // 2], dims=[1, 2])
    
    return torch.irfft(fi, signal_ndim=2, onesided=False, normalized=False)


class LowHighPassFilter(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.B = torch.distributions.bernoulli.Bernoulli(p)

    def forward(self, img):

        # half image size
        n = img.shape[-1] // 4

        if self.B.sample():
            return Filter(img, window_size=torch.randint(15, (1,))[0].item() + 10, high_pass=self.B.sample())
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + f'(Random low or high pass filter with proba p = {self.p})'