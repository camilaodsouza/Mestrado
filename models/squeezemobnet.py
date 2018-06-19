import math
import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ['SqueezeMobNet_', 'squeezemobnet_', 'squeezemobnet1_0', 'squeezemobnet1_1']


class Block_(nn.Module):
    """Depthwise conv + Pointwise conv"""
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block_, self).__init__()

        self.mob = nn.Sequential(
            OrderedDict([
                 ('pointwise', nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)),
                 #('BatchNorm', nn.BatchNorm2d(in_planes)),
                 ('activation', nn.ReLU(inplace=True)),
                 ('depthwise', nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)),
                 ('BatchNorm', nn.BatchNorm2d(out_planes)),
                 ('activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        out = self.mob(x)
        return out


class Fire_(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire_, self).__init__()
        self.inplanes = inplanes

        self.group1 = nn.Sequential(
            OrderedDict([
                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                ('BatchNorm', nn.BatchNorm2d(squeeze_planes)),
                ('squeeze_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                ('BatchNorm', nn.BatchNorm2d(expand1x1_planes)),
                ('expand1x1_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group3 = nn.Sequential(
            OrderedDict([
                ('expand3x3', Block_(squeeze_planes, expand3x3_planes))  # ,
                # ('BatchNorm', nn.BatchNorm2d(expand3x3_planes)),
                # ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x), self.group3(x)], 1)


class SqueezeMobNet_(nn.Module):

    def __init__(self, version=None, num_classes=10):
        super(SqueezeMobNet_, self).__init__()
        if version not in [1.0, 1.1, "cifar"]:
            raise ValueError("Unsupported SqueezeMobNet_ version {version}:"
                             "1.0, 1.1 or cifar expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            # Model to ImageNet dataset...
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire_(96, 16, 64, 64),
                Fire_(128, 16, 64, 64),
                Fire_(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire_(256, 32, 128, 128),
                Fire_(256, 48, 192, 192),
                Fire_(384, 48, 192, 192),
                Fire_(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire_(512, 64, 256, 256),
            )
        elif version == 1.1:
            # Model to ImageNet dataset...
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire_(64, 16, 64, 64),
                Fire_(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire_(128, 32, 128, 128),
                Fire_(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire_(256, 48, 192, 192),
                Fire_(384, 48, 192, 192),
                Fire_(384, 64, 256, 256),
                Fire_(512, 64, 256, 256),
            )
        elif version == "cifar":
            self.features = nn.Sequential(  # 32x32x3
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # 32x32x64
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 16x16x64
                Fire_(96, 32, 64, 64),  # 16x16x128
                Fire_(128, 16, 64, 64),  # 16x16x128
                Fire_(128, 32, 128, 128),  # 16x16x256
                Fire_(256, 32, 128, 128),  # 16x16x256
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 8x8x256
                Fire_(256, 48, 192, 192),  # 8x8x384
                Fire_(384, 48, 192, 192),  # 8x8x384
                Fire_(384, 64, 256, 256),  # 8x8x512
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 4x4x512
                Fire_(512, 64, 256, 256),  # 4x4x512
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        if version in [1.0, 1.1]:
            self.classifier = nn.Sequential(
                # nn.Dropout(p=0.5),
                final_conv,
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(13, stride=1)
            )
        elif version == "cifar":
            self.classifier = nn.Sequential(
                # nn.Dropout(p=0.5),
                final_conv,  # 4x4x10 for cifar10
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(4)  # 1x1x10 for cifar10
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is final_conv:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is final_conv:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1e-3)
                nn.init.constant_(module.bias, 0)
        """

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezemobnet_(**kwargs):
    model = SqueezeMobNet_(version="cifar", **kwargs)
    return model


#### ImageNet Model ####


def squeezemobnet1_0(**kwargs):
    model = SqueezeMobNet_(version=1.0, **kwargs)
    print(model)
    return model


def squeezemobnet1_1(**kwargs):
    model = SqueezeMobNet_(version=1.1, **kwargs)
    return model
