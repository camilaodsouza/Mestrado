import torch.nn as nn

__all__ = ['VGG', 'vgg11', 'vgg11bn', 'vgg13', 'vgg13bn', 'vgg16', 'vgg16bn', 'vgg19bn', 'vgg19']


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1/num_classes)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.classifier(x)
        return x


def make_layers(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # layers = layers[:-1]  # <<-- optional VGG desing number #1
    # layers += [nn.AvgPool2d(kernel_size=2, stride=2)]  # <<-- optional VGG desing number #1
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  # <<-- optional VGG desing number #2
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    return VGG(make_layers(cfg['A']), **kwargs)


def vgg11bn(**kwargs):
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(**kwargs):
    return VGG(make_layers(cfg['B']), **kwargs)


def vgg13bn(**kwargs):
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(**kwargs):
    return VGG(make_layers(cfg['D']), **kwargs)


def vgg16bn(**kwargs):
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(**kwargs):
    return VGG(make_layers(cfg['E']), **kwargs)


def vgg19bn(**kwargs):
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
