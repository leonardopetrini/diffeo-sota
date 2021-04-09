import torch
import torch.nn as nn


class swish(nn.Module):
    """
    swish activation. https://arxiv.org/abs/1710.05941
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class hswish(nn.Module):
    """
    h-swish activation. https://arxiv.org/abs/1905.02244
    """

    def __init__(self):
        super(hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class Drop_Connect:
    def __init__(self, drop_connect_rate):
        self.keep_prob = 1.0 - torch.tensor(drop_connect_rate, requires_grad=False)

    def __call__(self, x):
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + self.keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / self.keep_prob


def get_model_urls():
    model_urls = {
        'EfficientNetB0': '/home/lpetrini/data/efficientnet/efficientnet-b0-355c32eb.pth',
        'EfficientNetB1': '/home/lpetrini/data/efficientnet/efficientnet-b1-f1951068.pth',
        'EfficientNetB2': '/home/lpetrini/data/efficientnet/efficientnet-b2-8bb594d6.pth',
        'EfficientNetB3': '/home/lpetrini/data/efficientnet/efficientnet-b3-5fb5a3c3.pth',
        'EfficientNetB4': '/home/lpetrini/data/efficientnet/efficientnet-b4-6ed6700e.pth',
        'EfficientNetB5': '/home/lpetrini/data/efficientnet/efficientnet-b5-b6417697.pth',
        'EfficientNetB6': '/home/lpetrini/data/efficientnet/efficientnet-b6-c76e70fd.pth',
        'EfficientNetB7': '/home/lpetrini/data/efficientnet/efficientnet-b7-dcc49843.pth',
    }
    return model_urls