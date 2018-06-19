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
import torchvision
import torch
import numpy
from PIL import Image

dataset_names = ('cifar10', 'cifar100', 'mnist')

parser = argparse.ArgumentParser(description='Download Create Images')

parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')
parser.add_argument('-bs', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-gpu', '--gpu-id', default='1', type=str,
                    help='id for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# cuda device to use...
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

data_dir = os.path.join('.', args.dataset)
images_dir = os.path.join(data_dir, "images")
train_dir = os.path.join(images_dir, "train")
val_dir = os.path.join(images_dir, "val")

print(args.dataset)


def create_directories():
    os.makedirs(images_dir)
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    for i in range(nclasses):
        os.makedirs(os.path.join(train_dir, "class"+str(i)))
        os.makedirs(os.path.join(val_dir, "class"+str(i)))


def build_images(loader, dir):
    for batch_idx, (input, target) in enumerate(loader):
        print(input.size())
        print(type(input[0]))
        for i in range(input.size(0)):
            print(input[i].size())
            img_path = os.path.join(dir, "class" + str(target[i].item()), "image" + str(batch_idx * args.batch_size + i) + ".JPEG")
            torchvision.utils.save_image(input[i], img_path, padding=0)


"""
def build_images(set, dir, mode):
    if mode == "train":
        data = set.train_data
        labels = set.train_labels
    else:
        data = set.test_data
        labels = set.test_labels
    for i in range(len(data)):
        img = Image.fromarray(data[0])
        img = torchvision.transforms.ToTensor()(img)
        img_path = os.path.join(dir, "class" + str(labels[i]), "image" + str(i) + ".JPEG")
        torchvision.utils.save_image(img, img_path, padding=0)
"""


def mnist_build_images(loader, dir):
    for batch_idx, (input, target) in enumerate(loader):
        print(input.size())
        print(type(input[0]))
        for i in range(input.size(0)):
            mnist_save_image(input[i], os.path.join(
                dir, "class" + str(target[i].item()), "image" + str(batch_idx * args.batch_size + i) + ".png"))


def mnist_save_image(tensor, filename):
    from PIL import Image
    tensor = tensor.cpu()
    grid = torchvision.utils.make_grid(tensor, padding=0)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    new_ndarr = ndarr[:, :, 0]
    print(new_ndarr.shape)
    im = Image.fromarray(new_ndarr, mode='L')
    print("FILENAME:", filename)
    im.save(filename)  # Maybe bmp is not working for MNIST... Try png...


train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

if args.dataset == "cifar10":
    nclasses = 10
    create_directories()

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    build_images(train_loader, train_dir)
    build_images(test_loader, val_dir)
    #build_images(train_set, train_dir, "train")
    #build_images(test_set, val_dir, "test")

if args.dataset == "cifar100":
    nclasses = 100
    create_directories()

    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    build_images(train_loader, train_dir)
    build_images(test_loader, val_dir)


if args.dataset == "mnist":
    nclasses = 10
    create_directories()

    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    mnist_build_images(train_loader, train_dir)
    mnist_build_images(test_loader, val_dir)
