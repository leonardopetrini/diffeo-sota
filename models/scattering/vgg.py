import torch.nn as nn


class Scattering2dVGG(nn.Module):
    '''
        Simple CNN with 3x3 convs based on VGG
    '''
    def __init__(self, in_channels):
        super(Scattering2dVGG, self).__init__()
        self.in_channels = in_channels
        self.build()

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                self.in_channels = v

        layers += [nn.AdaptiveAvgPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.classifier =  nn.Linear(1024*4, 10)

    def forward(self, x):
        x = self.bn(x.view(-1, self.K, 8, 8))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x