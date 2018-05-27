import torch.nn as nn
# import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']

# model_urls = {
#    'alexnet': 'https://s3.amazonaws.com/pytorch/models/alexnet-owt-4df8aa71.pth',
# }


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        """
        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        else:
            num_classes = 10
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*3*3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 3 * 3)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(dataset):
    model = AlexNet(dataset)
    # if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
