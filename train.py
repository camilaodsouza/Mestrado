# insert this to the top of your scripts (usually main.py)
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#    traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

import argparse
import sys
import os
import time
import random
import numpy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import torchnet as tnt
import csv


from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import models
from folder import ImageFolder

cudnn.benchmark = False
cudnn.deterministic = True

numpy.set_printoptions(formatter={'float': '{:0.8f}'.format})
torch.set_printoptions(precision=8)
pd.set_option('display.width', 160)

dataset_names = ('mnist', 'cifar10', 'cifar100', 'imagenet2012')

local_model_names = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))
remote_model_names = sorted(name for name in torchvision_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('-dir', '--artifact-dir', type=str, metavar='DIR',
                    help='the project directory')
parser.add_argument('-data', '--dataset-dir', type=str, metavar='DATA',
                    help='output dir for logits extracted')
parser.add_argument('-x', '--executions', default=5, type=int, metavar='N',
                    help='Number of executions (default: 5)')
parser.add_argument('-d', '--dataset', metavar='DATA', default=None, choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names))
parser.add_argument('-lm', '--local-model', metavar='MODEL', default=None, choices=local_model_names,
                    help='model to be used: ' + ' | '.join(local_model_names))
parser.add_argument('-rm', '--remote-model', metavar='MODEL', default=None, choices=remote_model_names,
                    help='model to be used: ' + ' | '.join(remote_model_names))
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-bs', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-tss', '--train-set-split', default=None, type=float, metavar='TSS',
                    help='fraction of trainset to be used to validation')
parser.add_argument('-lr', '--original-learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('-lrdr', '--learning-rate-decay-rate', default=0.2, type=float, metavar='LRDR',
                    help='learning rate decay rate')
parser.add_argument('-lrdp', '--learning-rate-decay-period', default=30, type=int, metavar='LRDP',
                    help='learning rate decay period')
parser.add_argument('-lrde', '--learning-rate-decay-epochs', default="60 80 90", metavar='LRDE',
                    help='learning rate decay epochs')
parser.add_argument('-exps', '--experiments', default="baseline", type=str, metavar='EXPERIMENTS',
                    help='Experiments to be performed')
parser.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5*1e-4, type=float, metavar='W',
                    help='weight decay (default: 5*1e-4)')
parser.add_argument('-pf', '--print-freq', default=16, type=int, metavar='N',
                    help='print frequency (default: 16)')
parser.add_argument('-tr', '--train', const=True, nargs='?', type=bool,
                    help='if true, train the model')
parser.add_argument('-el', '--extract-logits', const=True, nargs='?', type=bool,
                    help='if true, extract logits')

args = parser.parse_args()
args.learning_rate_decay_epochs = sorted([int(item) for item in args.learning_rate_decay_epochs.split()])
args.experiments = args.experiments.split("_")

"""
raw_results = {
    'train_loss': [{} for item in range(args.epochs)],
    'train_entropy': [{} for item in range(args.epochs)],
    'train_acc1': [{} for item in range(args.epochs)],
    'val_entropy': [{} for item in range(args.epochs)],
    'val_acc1': [{} for item in range(args.epochs)]
}
"""


def execute():

    # Using seeds...
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if args.dataset == "mnist":
        args.number_of_dataset_classes = 10
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/mnist/images"
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        train_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == "cifar10":
        args.number_of_dataset_classes = 10
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/cifar10/images"
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == "cifar100":
        args.number_of_dataset_classes = 100
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/cifar100/images"
        normalize = transforms.Normalize(mean=[0.507, 0.486, 0.440], std=[0.267, 0.256, 0.276])
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        args.number_of_dataset_classes = 1000
        dataset_path = args.dataset_dir if args.dataset_dir else "../datasets/imagenet2012/images"
        if args.arch.startswith('inception'):
            size = (299, 299)
        else:
            size = (224, 256)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
            [transforms.RandomSizedCrop(size[0]),  # 224 , 299
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose(
            [transforms.Scale(size[1]),  # 256
             transforms.CenterCrop(size[0]),  # 224 , 299
             transforms.ToTensor(), normalize])

    # Defining the same normal classes for all experiments...
    args.normal_classes = sorted(random.sample(range(0, args.number_of_dataset_classes), args.number_of_model_classes))

    # Preparing paths...
    args.execution_path = os.path.join(args.experiment_path, "exec" + str(args.execution))
    if not os.path.exists(args.execution_path):
        os.makedirs(args.execution_path)

    # Preparing logger...
    writer = SummaryWriter(log_dir=args.execution_path)
    writer.add_text(str(vars(args)), str(vars(args)))

    # Printing args...
    print("\nARGUMENTS: ", vars(args))

    ###################
    # Preparing data...
    ###################

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # Creating sets and loaders...
    if args.train_set_split is None:
        train_set = ImageFolder(train_path, transform=train_transform, selected_classes=args.normal_classes,
                                target_transform=args.normal_classes.index)
        val_set = ImageFolder(val_path, transform=inference_transform, selected_classes=args.normal_classes,
                              target_transform=args.normal_classes.index)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                  pin_memory=True, shuffle=True, worker_init_fn=worker_init)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=True, shuffle=True, worker_init_fn=worker_init)
    else:
        train_set = ImageFolder(train_path, transform=train_transform, selected_classes=args.normal_classes,
                                target_transform=args.normal_classes.index)
        val_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes,
                              target_transform=args.normal_classes.index)

        train_sampler, val_sampler = compute_train_val_samplers(train_set, args.train_set_split)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                  pin_memory=True, sampler=train_sampler, worker_init_fn=worker_init)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=True, sampler=val_sampler, worker_init_fn=worker_init)

    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("TRAINSET LOADER SIZE: ====>>>> ", len(train_loader.sampler))
    print("VALIDSET LOADER SIZE: ====>>>> ", len(val_loader.sampler))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # Dataset created...
    print("\nDATASET:", args.dataset)

    # create model
    torch.manual_seed(args.execution)
    torch.cuda.manual_seed(args.execution)
    model = create_model()
    print("\nMODEL:", model)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    best_train_acc1 = 0
    best_val_acc1 = 0
    final_train_loss = None
    final_train_entropy = None
    final_val_entropy = None

    if args.train:
        ###################
        # Training...
        ###################

        # define loss function (criterion)...
        criterion = nn.CrossEntropyLoss().cuda()

        # define optimizer...
        optimizer = torch.optim.SGD(model.parameters(), lr=args.original_learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)

        # define optimizer...
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.original_learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)

        # define scheduler...
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.learning_rate_decay_epochs,
        #                                                  gamma=args.learning_rate_decay_rate)

        # define scheduler...
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.2, verbose=True)

        # model.initialize_parameters() ####### It works for AlexNet, LeNet and VGG...
        # initialize_parameters(model)

        print("\n################ TRAINING ################")
        best_model_file_path = os.path.join(args.execution_path, 'best_model.pth.tar')
        best_train_acc1, best_val_acc1, final_train_loss, final_train_entropy, final_val_entropy = \
            train_val(train_loader, val_loader, model, criterion, optimizer,
                      scheduler, args.epochs, writer, best_model_file_path)

        # save to json file
        writer.export_scalars_to_json(os.path.join(args.execution_path, 'log.json'))

    if args.extract_logits:
        ######################
        # Extracting logits...
        ######################

        print("\n################ EXTRACTING LOGITS ################")
        best_model_file_path = os.path.join(args.execution_path, 'best_model.pth.tar')

        # We need to use inference transform to extract... Even for trainset...
        if args.train_set_split is None:
            train_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes)
            val_set = ImageFolder(val_path, transform=inference_transform, selected_classes=args.normal_classes)
            test_set = ImageFolder(val_path, transform=inference_transform)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                      pin_memory=True, shuffle=True, worker_init_fn=worker_init)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                    pin_memory=True, shuffle=True, worker_init_fn=worker_init)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True, shuffle=True, worker_init_fn=worker_init)
        else:
            train_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes)
            val_set = ImageFolder(train_path, transform=inference_transform, selected_classes=args.normal_classes)
            test_set = ImageFolder(val_path, transform=inference_transform)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                      pin_memory=True, sampler=train_sampler, worker_init_fn=worker_init)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                    pin_memory=True, sampler=val_sampler, worker_init_fn=worker_init)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True, shuffle=True, worker_init_fn=worker_init)

        extract_logits_from_file(best_model_file_path, model, args.number_of_model_classes, args.execution_path,
                                 train_loader, val_loader, test_loader, "best_model")

    return best_val_acc1, best_train_acc1, final_train_loss, final_train_entropy, final_val_entropy


def train_val(train_loader, val_loader, model, criterion, optimizer, scheduler,
              total_epochs, writer, best_model_file_path):

    best_model_train_acc1 = -1
    best_model_val_acc1 = -1
    best_model_train_loss = None
    best_model_train_entropy = None
    best_model_val_entropy = None

    # for epoch in range(start_epoch, end_epoch + 1):
    for epoch in range(1, total_epochs + 1):
        print("\n######## EPOCH:", epoch, "OF", total_epochs, "########")

        # Adjusting learning rate (if not using reduce on plateau)...
        # scheduler.step()

        # Print current learning rate...
        for param_group in optimizer.param_groups:
            print("\nLEARNING RATE:\t", param_group["lr"])

        train_acc1, train_loss, train_entropy = train(train_loader, model, criterion, optimizer, epoch, writer)
        val_acc1, val_entropy = validate(val_loader, model, epoch, writer)

        """
        raw_results['train_acc1'][epoch][args.experiment] = train_acc1
        raw_results['train_loss'][epoch][args.experiment] = train_loss
        raw_results['train_entropy'][epoch][args.experiment] = train_entropy
        raw_results['val_acc1'][epoch][args.experiment] = val_acc1
        raw_results['val_entropy'][epoch][args.experiment] = val_entropy
        """

        # remember best acc1...
        # best_train_acc1 = max(train_acc1, best_train_acc1)
        # is_best = val_acc1 > best_val_acc1
        # best_val_acc1 = max(val_acc1, best_val_acc1)

        # if is_best:
        if val_acc1 > best_model_val_acc1:

            best_model_train_acc1 = train_acc1
            best_model_val_acc1 = val_acc1
            best_model_train_loss = train_loss
            best_model_train_entropy = train_entropy
            best_model_val_entropy = val_entropy

            print("!+NEW BEST+ {0:.3f} IN EPOCH {1}!!! SAVING... {2}\n".format(val_acc1, epoch, best_model_file_path))
            # torch.save({'epoch': epoch, 'arch': args.arch, 'state_dict': model.state_dict(),
            #            'best_val_acc1': best_val_acc1, 'optimizer': optimizer.state_dict()},
            #           best_model_file_path)
            full_state = {'epoch': epoch, 'arch': args.arch, 'model_state_dict': model.state_dict(),
                          # 'best_val_acc1': best_val_acc1, 'optimizer_state_dict': optimizer.state_dict()}
                          'best_val_acc1': best_model_val_acc1, 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(full_state, best_model_file_path)

        # print('$$$$ BEST: {0:.3f}\n'.format(best_val_acc1))
        print('!$$$$ BEST: {0:.3f}\n'.format(best_model_val_acc1))

        # Adjusting learning rate (if using reduce on plateau)...
        scheduler.step(val_acc1)

    # return best_train_acc1, best_val_acc1, train_loss, train_entropy, val_entropy
    return best_model_train_acc1, best_model_val_acc1, best_model_train_loss,\
        best_model_train_entropy, best_model_val_entropy


def compute_train_val_samplers(train_set_to_split, split_fraction):
    # Preparing train and validation samplers...
    total_examples = {}
    for index in range(len(train_set_to_split)):
        _, label = train_set_to_split[index]
        if label not in total_examples:
            total_examples[label] = 1
        else:
            total_examples[label] += 1
    train_indexes = []
    val_indexes = []
    train_indexes_count = {}
    val_indexes_count = {}
    indexes_count = {}
    for index in range(len(train_set_to_split)):
        _, label = train_set_to_split[index]
        if label not in indexes_count:
            indexes_count[label] = 1
            train_indexes.append(index)
            train_indexes_count[label] = 1
            val_indexes_count[label] = 0
        else:
            indexes_count[label] += 1
            # if indexes_count[label] <= int(total_examples[label] * args.train_set_split):
            if indexes_count[label] <= int(total_examples[label] * split_fraction):
                train_indexes.append(index)
                train_indexes_count[label] += 1
            else:
                val_indexes.append(index)
                val_indexes_count[label] += 1
    print("TRAIN SET INDEXES TOTALS:", train_indexes_count)
    print("VALID SET INDEXES TOTALS:", val_indexes_count)
    # train_sampler = SubsetRandomSampler(train_indexes)
    # val_sampler = SubsetRandomSampler(val_indexes)
    return SubsetRandomSampler(train_indexes), SubsetRandomSampler(val_indexes)


def compute_entropy(variable, dim=1):
    softmax = nn.Softmax(dim=dim)
    logsoftmax = nn.LogSoftmax(dim=dim)
    softmax_output = softmax(variable)  # .data)#.data
    logsoftmax_output = logsoftmax(variable)  # .data)#.data
    return -(softmax_output * logsoftmax_output).sum(dim=dim)


def worker_init(worker_id):
    # random.seed(args.execution)
    random.seed(0)


def create_model():
    if args.dataset in ["mnist", "cifar10", "cifar100"]:
        # arch = args.local_model
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:  # args.dataset == "imagenet2012":
        # arch = args.remote_model
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def extract_logits_from_file(model_file, model, number_of_classes, path,
                             train_loader, val_loader, test_loader, suffix):

    if os.path.isfile(model_file):
        print("\n=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(model_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        return

    logits_train_file = '{}/{}/{}.pth'.format(path, 'logits', suffix + '_train')
    logits_val_file = '{}/{}/{}.pth'.format(path, 'logits', suffix + '_val')
    logits_test_file = '{}/{}/{}.pth'.format(path, 'logits', suffix + '_test')

    # logits_train_set = extract_logits(model, number_of_classes, train_loader, logits_train_file)
    extract_logits(model, number_of_classes, train_loader, logits_train_file)
    # logits_val_set = extract_logits(model, number_of_classes, val_loader, logits_val_file)
    extract_logits(model, number_of_classes, val_loader, logits_val_file)
    # logits_test_set = extract_logits(model, number_of_classes, test_loader, logits_test_file)
    extract_logits(model, number_of_classes, test_loader, logits_test_file)


def extract_logits(model, number_of_classes, loader, path):

    # print('\nExtract logits on {}set'.format(loader.dataset.set))
    print('Extract logits on {}'.format(loader.dataset))

    logits = torch.Tensor(len(loader.sampler), number_of_classes)
    # logits = torch.Tensor(len(loader.sampler), len(loader.dataset.classes))
    targets = torch.Tensor(len(loader.sampler))
    print("LOGITS:\t\t", logits.size())
    print("TARGETS:\t", targets.size())

    # switch to evaluate mode
    model.eval()

    for batch_id, batch in enumerate(tqdm(loader)):
        img = batch[0]
        # target = batch[2]
        target = batch[1]
        current_bsize = img.size(0)
        from_ = int(batch_id * loader.batch_size)
        to_ = int(from_ + current_bsize)

        img = img.cuda(async=True)

        input_var = Variable(img, requires_grad=False)
        output = model(input_var)

        logits[from_:to_] = output.data.cpu()
        targets[from_:to_] = target

    os.system('mkdir -p {}'.format(os.path.dirname(path)))
    print('save ' + path)
    torch.save((logits, targets), path)
    print('')
    return logits, targets


def train(train_loader, model, criterion, optimizer, epoch, writer):
    # Meters...
    train_loss = tnt.meter.AverageValueMeter()
    train_entropy = tnt.meter.AverageValueMeter()
    train_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    train_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to train mode
    model.train()

    # Start timer...
    train_batch_start_time = time.time()

    for batch_index, (input_tensor, target_tensor) in enumerate(train_loader):
        batch_index += 1

        # measure data loading time
        train_data_time = time.time() - train_batch_start_time

        input_var = torch.autograd.Variable(input_tensor)
        target_tensor = target_tensor.cuda(async=True)
        target_var = torch.autograd.Variable(target_tensor)

        # compute output
        output_var = model(input_var)

        # Working with entropy...
        entropy = compute_entropy(output_var, dim=1)
        mean_entropy = entropy.sum()/entropy.size(0)

        # compute loss
        if args.regularization_type == "l2":
            # print("L2 REGULARIZATION\t", args.regularization_value)
            loss_var = criterion(output_var, target_var) + (args.regularization_value * torch.norm(output_var, 2))
        elif args.regularization_type == "ne":
            # print("NEGATIVE ENTROPIC REGULARIZATION\t", args.regularization_value)
            loss_var = criterion(output_var, target_var) - (args.regularization_value * mean_entropy)
        elif args.regularization_type == "pie":
            # print("POSITIVE INVERTED ENTROPIC REGULARIZATION\t", args.regularization_value)
            loss_var = criterion(output_var, target_var) + (args.regularization_value / mean_entropy)
        else:
            # print("NO REGULARIZATION")
            loss_var = criterion(output_var, target_var)

        # accumulate metrics over epoch
        train_loss.add(loss_var.data[0])
        train_entropy.add(mean_entropy.data[0])
        train_acc.add(output_var.data, target_var.data)
        train_conf.add(output_var.data, target_var.data)

        # zero grads, compute gradients and do optimizer step
        optimizer.zero_grad()
        loss_var.backward()
        optimizer.step()

        # measure elapsed time
        train_batch_time = time.time() - train_batch_start_time

        if batch_index % args.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Data {train_data_time:.6f}\t'
                  'Time {train_batch_time:.6f}\t'
                  'Loss {loss:.4f}\t'
                  'Entropy {entropy:.4f}\t\t'
                  'Acc1 {acc1_meter:.2f}\t'
                  'Acc5 {acc5_meter:.2f}'
                  .format(epoch, batch_index, len(train_loader),
                          train_data_time=train_data_time,
                          train_batch_time=train_batch_time,
                          loss=train_loss.value()[0],
                          entropy=train_entropy.value()[0],
                          acc1_meter=train_acc.value()[0],
                          acc5_meter=train_acc.value()[1],
                          )
                  )

        # Restart timer...
        train_batch_start_time = time.time()

    print("\nEPOCH TRAIN RESULTS")
    print("LOSS:\t\t", train_loss.value())
    print("ENTROPY:\t", train_entropy.value())
    print("ACCURACY:\t", train_acc.value())
    print("\nCONFUSION:\n", train_conf.value())

    writer.add_scalar('train/loss', train_loss.value()[0], epoch)
    writer.add_scalar('train/entropy', train_entropy.value()[0], epoch)
    writer.add_scalar('train/acc1', train_acc.value()[0], epoch)
    writer.add_scalar('train/acc5', train_acc.value()[1], epoch)

    print('\n#### TRAIN: {acc1:.3f}\n\n'.format(acc1=train_acc.value()[0]))
    # confusion = dict(np.ndenumerate(train_conf.value()))
    # confusion = {str(key):float(value) for key,value in confusion.items()}
    # writer.add_scalars('train/confusion', confusion, epoch)

    return train_acc.value()[0], train_loss.value()[0], train_entropy.value()[0]


def validate(val_loader, model, epoch, writer):
    # Meters...
    val_entropy = tnt.meter.AverageValueMeter()
    val_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    val_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    # Start timer...
    val_batch_start_time = time.time()

    for batch_index, (input_tensor, target_tensor) in enumerate(val_loader):
        batch_index += 1

        # measure data loading time
        val_data_time = time.time()-val_batch_start_time

        input_var = torch.autograd.Variable(input_tensor, volatile=True)
        target_tensor = target_tensor.cuda(async=True)
        target_var = torch.autograd.Variable(target_tensor, volatile=True)

        # compute output
        output_var = model(input_var)

        # Working with entropy...
        entropy = compute_entropy(output_var, dim=1)
        mean_entropy = entropy.data.sum()/entropy.data.size(0)

        # accumulate metrics over epoch
        val_entropy.add(mean_entropy)
        val_acc.add(output_var.data, target_var.data)
        val_conf.add(output_var.data, target_var.data)

        # measure elapsed time
        val_batch_time = time.time()-val_batch_start_time

        if batch_index % args.print_freq == 0:
            print('Valid Epoch: [{0}][{1}/{2}]\t'
                  'Data {val_data_time:.6f}\t'
                  'Time {val_batch_time:.6f}\t'
                  'Entropy {entropy:.4f}\t\t'
                  'Acc1 {acc1_meter:.2f}\t'
                  'Acc5 {acc5_meter:.2f}'
                  .format(epoch, batch_index, len(val_loader),
                          val_data_time=val_data_time,
                          val_batch_time=val_batch_time,
                          entropy=val_entropy.value()[0],
                          acc1_meter=val_acc.value()[0],
                          acc5_meter=val_acc.value()[1],
                          )
                  )

        # Restart timer...
        val_batch_start_time = time.time()

    print("\nEPOCH VALID RESULTS")
    print("ENTROPY:\t", val_entropy.value())
    print("ACCURACY:\t", val_acc.value())
    print("\nCONFUSION:\n", val_conf.value())

    writer.add_scalar('val/entropy', val_entropy.value()[0], epoch)
    writer.add_scalar('val/acc1', val_acc.value()[0], epoch)
    writer.add_scalar('val/acc5', val_acc.value()[1], epoch)

    print('\n#### VALID: {acc1:.3f}\n'.format(acc1=val_acc.value()[0]))
    # confusion = dict(np.ndenumerate(val_conf.value()))
    # confusion = {str(key):float(value) for key,value in confusion.items()}
    # writer.add_scalars('val/confusion', confusion, epoch)

    return val_acc.value()[0], val_entropy.value()[0]


def main():

    if not (args.train or args.extract_logits):
        print("\nNOTHING TO DO!!!\n")
        sys.exit()

    all_acc1_stats = {}

    for experiment in args.experiments:

        print("\n\n")
        print("****************************************************************")
        print("EXPERIMENT:", experiment.upper())
        print("****************************************************************")

        acc1_results = {}
        acc1_stats = pd.DataFrame()

        if args.local_model is not None:
            args.arch = args.local_model
        else:
            args.arch = args.remote_model

        args.experiment_path = os.path.join("artifacts", args.dataset, args.arch, experiment)
        print("\nEXPERIMENT PATH:", args.experiment_path)

        experiment_configs = experiment.split("+")

        for config in experiment_configs:
            config = config.split("~")
            if config[0] == "nmc":
                args.number_of_model_classes = int(config[1])
                print("NUMBER OF MODEL CLASSES:", args.number_of_model_classes)
            elif config[0] == "rt":
                args.regularization_type = str(config[1])
                print("REGULARIZATION TYPE:", args.regularization_type)
            if config[0] == "rv":
                args.regularization_value = float(config[1])
                print("REGULARIZATION VALUE:", args.regularization_value)

        for execution in range(1, args.executions + 1):

            # Using seeds
            args.execution = execution
            # random.seed(args.execution)
            # numpy.random.seed(args.execution)
            # torch.manual_seed(args.execution)
            # torch.cuda.manual_seed(args.execution)

            print("\n################ EXECUTION:", args.execution, "OF", args.executions, "################")

            # execute experiment...
            val_acc1, train_acc1, train_loss, train_entropy, val_entropy = execute()

            # results and auc_statistics...
            (acc1_results["BEST VALID [ACC1]"], acc1_results["BEST TRAIN [ACC1]"],
             acc1_results["TRAIN LOSS"], acc1_results["TRAIN ENTROPY"], acc1_results["VALID ENTROPY"]) =\
                (val_acc1, train_acc1,
                 train_loss, train_entropy, val_entropy)

            # appending results...
            acc1_stats = acc1_stats.append(acc1_results, ignore_index=True)
            acc1_stats = acc1_stats[["BEST VALID [ACC1]", "BEST TRAIN [ACC1]",
                                     "TRAIN LOSS", "TRAIN ENTROPY", "VALID ENTROPY"]]

        # print("\n################################\n", "EXPERIMENT STATISTICS", "\n################################\n")
        # print("\n", experiment.upper())
        # print("\n", acc1_stats.transpose())
        # print("\n", acc1_stats.describe())

        all_acc1_stats[experiment] = acc1_stats

        """
        csv.register_dialect('unixpwd', delimiter='\t', quoting=csv.QUOTE_NONE)
        for key in raw_results:
            file_path = os.join(args.experiment_path, key + '.csv') 
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = [str(item) for item in range(args.executions)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='unixpwd')
                writer.writeheader()
                for epoch in raw_results[key]:
                    writer.writerow(raw_results[key][epoch])
        """

    print("\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", "OVERALL STATISTICS", "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    for key in all_acc1_stats:
        print("\n", key.upper())
        print("\n", all_acc1_stats[key].transpose())
        print("\n", all_acc1_stats[key].describe())

    print("\n\n\n")


if __name__ == '__main__':
    main()
