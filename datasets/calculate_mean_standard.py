# insert this to the top of your scripts (usually main.py)
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#     traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

import argparse
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from folder import ImageFolder

dataset_names = ('cifar10', 'cifar100', 'mnist')

parser = argparse.ArgumentParser(description='Calculate Mean Standard')

parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')

args = parser.parse_args()

data_dir = os.path.join('.', args.dataset)

print(args.dataset)

train_transform = transforms.Compose([transforms.ToTensor()])

if args.dataset == "cifar10":
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    print(type(train_set.train_data))
    print(train_set.train_data.shape)
    print(train_set.__getitem__(0)[0].size())
    print(train_set.train_data.mean(axis=(0, 1, 2))/255)
    print(train_set.train_data.std(axis=(0, 1, 2))/255)
    train_set = ImageFolder("cifar10/images/train", transform=train_transform)
    image_tensor = torch.stack([x for x, y in train_set])
    image_tensor = image_tensor.numpy()
    print(image_tensor.shape)
    print(train_set.__getitem__(0)[0].size())
    print(image_tensor.mean(axis=(0, 2, 3)))
    print(image_tensor.std(axis=(0, 2, 3)))

elif args.dataset == "cifar100":
    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    print(train_set.train_data.shape)
    print(np.mean(train_set.train_data, axis=(0, 1, 2))/255)
    print(np.std(train_set.train_data, axis=(0, 1, 2))/255)

elif args.dataset == "mnist":
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    print(list(train_set.train_data.size()))
    print(train_set.train_data.float().mean()/255)
    print(train_set.train_data.float().std()/255)
