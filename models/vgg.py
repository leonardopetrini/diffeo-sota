'''VGG11/13/16/19 in Pytorch.'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_ch=3, num_classes=10, batch_norm=True, pooling='max', pooling_size=2, param_list=False):
        super(VGG, self).__init__()
        if pooling == True:
            pooling = 'max'
        self.features = self._make_layers(cfg[vgg_name], ch=num_ch, bn=batch_norm, pooling=pooling, ps=pooling_size, param_list=param_list)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, ch, bn, pooling, ps, param_list):
        layers = []
        in_channels = ch
        if ch == 1:
            layers.append(nn.ZeroPad2d(2))
        if param_list:
            convLayer = Conv2dList
        else:
            convLayer = nn.Conv2d
        for x in cfg:
            if x == 'M':
                if pooling == 'max':
                    layers += [nn.MaxPool2d(kernel_size=ps, stride=2, padding=ps // 2 + ps % 2 - 1)]
                elif pooling == 'avg':
                    layers += [nn.AvgPool2d(kernel_size=ps, stride=2, padding=ps // 2 + ps % 2 - 1)]
                else:
                    layers += [SubSampling(kernel_size=ps, stride=2)]
            else:
                if bn:
                    layers += [convLayer(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [convLayer(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class Conv2dList(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='constant'):
        super().__init__()

        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if bias is not None:
            bias = nn.Parameter(torch.empty(out_channels, ))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        n = max(1, 128 * 256 // (in_channels * kernel_size * kernel_size))
        weight = nn.ParameterList([nn.Parameter(weight[j: j + n]) for j in range(0, len(weight), n)])

        setattr(self, 'weight', weight)
        setattr(self, 'bias', bias)

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride

    def forward(self, x):

        weight = self.weight
        if isinstance(weight, nn.ParameterList):
            weight = torch.cat(list(self.weight))

        return F.conv2d(x, weight, self.bias, self.stride, self.padding)


class SubSampling(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(SubSampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[..., 0, 0]



# '''VGG11/13/16/19 in Pytorch.'''
# import torch
# import torch.nn as nn
#
#
# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
#
# class VGG(nn.Module):
#     def __init__(self, vgg_name, num_ch=3, num_classes=10, batch_norm=True, pooling=True, pooling_size=2):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name], ch=num_ch, bn=batch_norm, pooling=pooling, ps=pooling_size)
#         self.classifier = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#     def _make_layers(self, cfg, ch, bn, pooling, ps):
#         layers = []
#         in_channels = ch
#         for x in cfg:
#             if x == 'M':
#                 if pooling:
#                     layers += [nn.MaxPool2d(kernel_size=ps, stride=2, padding=ps // 2 + ps % 2 - 1)]
#                 else:
#                     layers += [SubSampling(kernel_size=2, stride=2)]
#             else:
#                 if bn:
#                     layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                                nn.BatchNorm2d(x),
#                                nn.ReLU(inplace=True)]
#                 else:
#                     layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                                nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)
#
#
# def test():
#     net = VGG('VGG11')
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())
#
# # test()
#
# class SubSampling(nn.Module):
#     def __init__(self, kernel_size, stride=None):
#         super(SubSampling, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride if (stride is not None) else kernel_size
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return input.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[..., 0, 0]
