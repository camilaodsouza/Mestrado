# insert this to the top of your scripts (usually main.py)
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#    traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

import argparse
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
import torchvision
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import torchnet as tnt
import csv


from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.modules.module import _addindent
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import models
from datasets import ImageFolder

cudnn.benchmark = False
cudnn.deterministic = True
#cudnn.benchmark = True
#cudnn.deterministic = False

numpy.set_printoptions(formatter={'float': '{:0.4f}'.format})
torch.set_printoptions(precision=4)
pd.set_option('display.width', 160)

dataset_names = ('mnist', 'cifar10', 'cifar100', 'imagenet2012')

local_model_names = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))
remote_model_names = sorted(name for name in torchvision_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('-dd', '--dataset-dir', type=str, metavar='DATA',
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
parser.add_argument('-e', '--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-bs', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-tss', '--train-set-split', default=None, type=float, metavar='TSS',
                    help='fraction of trainset to be used to validation')
parser.add_argument('-lr', '--original-learning-rate', default=0.05, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('-lrdr', '--learning-rate-decay-rate', default=0.2, type=float, metavar='LRDR',
                    help='learning rate decay rate')
parser.add_argument('-lrde', '--learning-rate-decay-epochs', default="60 90 110", metavar='LRDE',
                    help='learning rate decay epochs')
parser.add_argument('-lrdp', '--learning-rate-decay-period', default=30, type=int, metavar='LRDP',
                    help='learning rate decay period')
parser.add_argument('-exps', '--experiments', default="baseline", type=str, metavar='EXPERIMENTS',
                    help='Experiments to be performed')
parser.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5*1e-4, type=float, metavar='W',
                    help='weight decay (default: 5*1e-4)')
parser.add_argument('-pf', '--print-freq', default=16, type=int, metavar='N',
                    help='print frequency (default: 16)')
parser.add_argument('-gpu', '--gpu-id', default='0', type=str,
                    help='id for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
args.learning_rate_decay_epochs = sorted([int(item) for item in args.learning_rate_decay_epochs.split()])
args.experiments = args.experiments.split("_")

if args.local_model is not None:
    args.arch = args.local_model
else:
    args.arch = args.remote_model

# cuda device to be used...
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# RAW_RESULTS IS CURRENTLY A GLOBAL VARIABLE...
raw_results = {
    'train_loss': [{} for item in range(args.epochs)],
    'train_entropy': [{} for item in range(args.epochs)],
    'train_acc1': [{} for item in range(args.epochs)],
    'val_entropy': [{} for item in range(args.epochs)],
    'val_acc1': [{} for item in range(args.epochs)]
}


def execute():

    ######################################
    # Initial configuration...
    ######################################

    # Using seeds...
    random.seed(args.base_seed)
    numpy.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed(args.base_seed)
    args.execution_seed = args.base_seed + args.execution
    print("EXECUTION SEED:", args.execution_seed)

    # Configuring args and dataset...
    if args.dataset == "mnist":
        args.number_of_dataset_classes = 10
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "datasets/mnist"
        train_set = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=train_transform)
        val_set = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=inference_transform)
    elif args.dataset == "cifar10":
        args.number_of_dataset_classes = 10
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
        normalize = transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "datasets/cifar10"
        train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
        val_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=inference_transform)
    elif args.dataset == "cifar100":
        args.number_of_dataset_classes = 100
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 100
        normalize = transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose([transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "datasets/cifar100"
        train_set = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=train_transform)
        val_set = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=inference_transform)
    else:
        args.number_of_dataset_classes = 1000
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 1000

        # Relevant for training on ImageNet...
        if args.arch.startswith("squeezenet"):
            args.original_learning_rate = 0.01
            args.weight_decay = 2 * 1e-4
        if args.arch.startswith("squeezemobnet"):
            args.original_learning_rate = 0.01
            args.weight_decay = 2 * 1e-4
        if args.arch.startswith("mobile"):
            args.original_learning_rate = 0.05
            args.weight_decay = 4 * 1e-5

        if args.arch.startswith('inception'):
            size = (299, 299)
        else:
            size = (224, 256)

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(size[0]),  # 224 , 299
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])
        inference_transform = transforms.Compose(
            [transforms.Resize(size[1]),  # 256
             transforms.CenterCrop(size[0]),  # 224 , 299
             transforms.ToTensor(), normalize])
        dataset_path = args.dataset_dir if args.dataset_dir else "/mnt/ssd/imagenet_scripts/2012/images"
        train_path = os.path.join(dataset_path, 'train')
        val_path = os.path.join(dataset_path, 'val')
        train_set = ImageFolder(train_path, transform=train_transform)
        val_set = ImageFolder(val_path, transform=inference_transform)

    # Preparing paths...
    args.execution_path = os.path.join(args.experiment_path, "exec" + str(args.execution))
    if not os.path.exists(args.execution_path):
        os.makedirs(args.execution_path)

    # Printing args...
    print("\nARGUMENTS: ", vars(args))

    # Preparing logger...
    writer = SummaryWriter(log_dir=args.execution_path)
    writer.add_text(str(vars(args)), str(vars(args)))

    ######################################
    # Preparing data...
    ######################################

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, worker_init_fn=worker_init)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, worker_init_fn=worker_init)

    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("TRAINSET LOADER SIZE: ====>>>> ", len(train_loader.sampler))
    print("VALIDSET LOADER SIZE: ====>>>> ", len(val_loader.sampler))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # Dataset created...
    print("\nDATASET:", args.dataset)

    # create model
    torch.manual_seed(args.execution_seed)
    torch.cuda.manual_seed(args.execution_seed)
    print("=> creating model '{}'".format(args.arch))
    # model = create_model()
    model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
    model.cuda()
    print("\nMODEL:", model)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed(args.base_seed)

    #########################################
    # Training...
    #########################################

    # define loss function (criterion)...
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer...
    optimizer = torch.optim.SGD(model.parameters(), lr=args.original_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters())  # , lr=0.01)  # , weight_decay=2e-4)

    # define scheduler...
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.2, verbose=True,
    #                                                       threshold=0.05, threshold_mode='rel')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.learning_rate_decay_epochs,
                                                     gamma=args.learning_rate_decay_rate)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

    # model.initialize_parameters() ####### It works for AlexNet_, LeNet and VGG...
    # initialize_parameters(model)

    print("\n################ TRAINING ################")
    best_model_file_path = os.path.join(args.execution_path, 'best_model.pth.tar')
    best_train_acc1, best_val_acc1, final_train_loss, final_train_entropy, final_val_entropy = \
        train_val(train_loader, val_loader, model, criterion, optimizer,
                  scheduler, args.epochs, writer, best_model_file_path)

    # save to json file
    writer.export_scalars_to_json(os.path.join(args.execution_path, 'log.json'))

    ########################################################################################
    # Computing inference times and number of parameters...
    ########################################################################################

    print("\n################ COMPUTING INFERENCE TIMES AND MODEL WEIGHTS, BIAS AND PARAMETERS ################")

    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.workers, shuffle=True, worker_init_fn=worker_init)

    mean_cpu_inference_time = 1000 * compute_total_inference_time(model, val_loader, "cpu") / len(val_loader.sampler)
    mean_gpu_inference_time = 1000 * compute_total_inference_time(model, val_loader, "gpu") / len(val_loader.sampler)

    print("\nMEAN CPU INFERENCE TIME (MILISECONDS):\t{:.4f}".format(mean_cpu_inference_time))
    print("MEAN GPU INFERENCE TIME (MILISECONDS):\t{:.4f}".format(mean_gpu_inference_time))
    print("\nNUMBER OF WEIGHTS, BIAS AND PARAMETERS:")
    print(torch_summarize(model), "\n")

    return (best_train_acc1, best_val_acc1, final_train_loss, final_train_entropy, final_val_entropy,
            mean_cpu_inference_time, mean_gpu_inference_time, )


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
        scheduler.step()

        # Print current learning rate...
        for param_group in optimizer.param_groups:
            print("\nLEARNING RATE:\t", param_group["lr"])

        train_acc1, train_loss, train_entropy = train(train_loader, model, criterion, optimizer, epoch, writer)
        val_acc1, val_entropy = validate(val_loader, model, epoch, writer)

        # Saving raw results...
        raw_results['train_acc1'][epoch - 1][args.execution] = train_acc1
        raw_results['train_loss'][epoch - 1][args.execution] = train_loss
        raw_results['train_entropy'][epoch - 1][args.execution] = train_entropy
        raw_results['val_acc1'][epoch - 1][args.execution] = val_acc1
        raw_results['val_entropy'][epoch - 1][args.execution] = val_entropy

        # if is best...
        if val_acc1 > best_model_val_acc1:

            best_model_train_acc1 = train_acc1
            best_model_val_acc1 = val_acc1
            best_model_train_loss = train_loss
            best_model_train_entropy = train_entropy
            best_model_val_entropy = val_entropy

            print("!+NEW BEST+ {0:.3f} IN EPOCH {1}!!! SAVING... {2}\n".format(val_acc1, epoch, best_model_file_path))
            full_state = {'epoch': epoch, 'arch': args.arch, 'model_state_dict': model.state_dict(), 'best_val_acc1': best_model_val_acc1}
            torch.save(full_state, best_model_file_path)

        print('!$$$$ BEST: {0:.3f}\n'.format(best_model_val_acc1))

        # Adjusting learning rate (if using reduce on plateau)...
        #### scheduler.step(val_acc1)
        #scheduler.step(train_loss)

    return (best_model_train_acc1, best_model_val_acc1, best_model_train_loss,
            best_model_train_entropy, best_model_val_entropy)


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

        # moving to GPU...
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda(non_blocking=True)

        # compute output
        output_tensor = model(input_tensor)

        # Working with entropy...
        entropy = compute_entropies(output_tensor, dim=1)
        mean_entropy = entropy.sum()/entropy.size(0)

        # compute loss
        if args.regularization_type == "l2":
            loss = criterion(output_tensor, target_tensor) + (args.regularization_value * torch.norm(output_tensor, 2))
        elif args.regularization_type == "ne":
            loss = criterion(output_tensor, target_tensor) - (args.regularization_value * mean_entropy)
        elif args.regularization_type == "pie":
            loss = criterion(output_tensor, target_tensor) + (args.regularization_value / mean_entropy)
        else:
            loss = criterion(output_tensor, target_tensor)

        # accumulate metrics over epoch
        train_loss.add(loss.item())
        train_entropy.add(mean_entropy.item())
        train_acc.add(output_tensor.detach(), target_tensor.detach())
        train_conf.add(output_tensor.detach(), target_tensor.detach())

        # zero grads, compute gradients and do optimizer step
        optimizer.zero_grad()
        loss.backward()
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

    print("\nCONFUSION:\n", train_conf.value())
    print('\n#### TRAIN: {acc1:.3f}\n\n'.format(acc1=train_acc.value()[0]))

    # confusion = dict(np.ndenumerate(train_conf.value()))
    # confusion = {str(key):float(value) for key,value in confusion.items()}
    # writer.add_scalars('train/confusion', confusion, epoch)
    writer.add_scalar('train/loss', train_loss.value()[0], epoch)
    writer.add_scalar('train/entropy', train_entropy.value()[0], epoch)
    writer.add_scalar('train/acc1', train_acc.value()[0], epoch)
    writer.add_scalar('train/acc5', train_acc.value()[1], epoch)

    return train_acc.value()[0], train_loss.value()[0], train_entropy.value()[0]


def validate(val_loader, model, epoch, writer):
    # Meters...
    val_entropy = tnt.meter.AverageValueMeter()
    val_acc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    val_conf = tnt.meter.ConfusionMeter(args.number_of_model_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    #correct = 0
    #total = 0

    # Start timer...
    val_batch_start_time = time.time()

    with torch.no_grad():

        for batch_index, (input_tensor, target_tensor) in enumerate(val_loader):
            batch_index += 1

            # measure data loading time
            val_data_time = time.time()-val_batch_start_time

            """
            input_tensor = torch.autograd.Variable(input_tensor, volatile=True)
            target_tensor = target_tensor.cuda(async=True)
            target_tensor = torch.autograd.Variable(target_tensor, volatile=True)
            """

            # moving to GPU...
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda(non_blocking=True)

            # compute output
            output_tensor = model(input_tensor)

            # Working with entropy...
            entropy = compute_entropies(output_tensor, dim=1)
            mean_entropy = entropy.sum()/entropy.size(0)

            # accumulate metrics over epoch
            val_entropy.add(mean_entropy.item())
            val_acc.add(output_tensor.detach(), target_tensor.detach())
            val_conf.add(output_tensor.detach(), target_tensor.detach())

            # measure elapsed time
            val_batch_time = time.time()-val_batch_start_time

            #_, predicted = output_tensor.max(1)
            #total += target_tensor.size(0)
            #correct += predicted.eq(target_tensor).sum().item()
            #print("CHECK:", 100. * correct / total)

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

    print("\nCONFUSION:\n", val_conf.value())
    print('\n#### VALID: {acc1:.3f}\n'.format(acc1=val_acc.value()[0]))

    # confusion = dict(np.ndenumerate(val_conf.value()))
    # confusion = {str(key):float(value) for key,value in confusion.items()}
    # writer.add_scalars('val/confusion', confusion, epoch)
    writer.add_scalar('val/entropy', val_entropy.value()[0], epoch)
    writer.add_scalar('val/acc1', val_acc.value()[0], epoch)
    writer.add_scalar('val/acc5', val_acc.value()[1], epoch)

    return val_acc.value()[0], val_entropy.value()[0]


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
    train_sampler = SubsetRandomSampler(train_indexes)
    val_sampler = SubsetRandomSampler(val_indexes)
    # return SubsetRandomSampler(train_indexes), SubsetRandomSampler(val_indexes)
    return train_sampler, val_sampler


def compute_entropies(tensor, dim=1):
    softmax = nn.Softmax(dim=dim)
    logsoftmax = nn.LogSoftmax(dim=dim)
    softmax_output = softmax(tensor)  # .data)#.data
    logsoftmax_output = logsoftmax(tensor)  # .data)#.data
    return -(softmax_output * logsoftmax_output).sum(dim=dim)


def worker_init(worker_id):
    random.seed(args.base_seed)


def create_model():

    if args.dataset in ["mnist", "cifar10", "cifar100"]:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.number_of_model_classes)
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model


def extract_logits_from_file(model_file, model, number_of_classes, path, train_loader, val_loader, test_loader, suffix):

    # Loading best model...
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

    extract_logits(model, number_of_classes, train_loader, logits_train_file)
    extract_logits(model, number_of_classes, val_loader, logits_val_file)
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

    with torch.no_grad():

        for batch_id, (input_tensor, target_tensor) in enumerate(tqdm(loader)):
            # input_tensor = batch[0]
            # target_tensor = batch[1]

            # moving to GPU...
            input_tensor = input_tensor.cuda()
            # target_tensor = target_tensor.cuda(non_blocking=True)

            # compute output
            output = model(input_tensor)

            current_bsize = input_tensor.size(0)
            from_ = int(batch_id * loader.batch_size)
            to_ = int(from_ + current_bsize)

            # logits[from_:to_] = output.data.cpu()
            logits[from_:to_] = output.cpu()
            targets[from_:to_] = target_tensor

    os.system('mkdir -p {}'.format(os.path.dirname(path)))
    print('save ' + path)
    torch.save((logits, targets), path)
    print('')
    return logits, targets


def compute_total_inference_time(model, val_loader, mode):

    total_inference_time = 0

    if mode == "cpu":
        model.cpu()
    elif mode == "gpu":
        model.cuda()

    print("\nCOMPUTING INFERENCE TIME: {}".format(mode.upper()))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        for input_tensor, _ in tqdm(val_loader):

            if mode == "cpu":
                input_tensor = input_tensor.cpu()
            elif mode == "gpu":
                input_tensor = input_tensor.cuda()

            # compute output
            initial_time = time.time()
            _ = model(input_tensor)
            final_time = time.time()
            instance_inference_time = final_time - initial_time
            # instance_inference_time = timeit.timeit("model(input_tensor)", number=1)

            total_inference_time += instance_inference_time

    return total_inference_time


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'

    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([numpy.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def main():

    overall_stats = {}

    for experiment in args.experiments:

        print("\n\n")
        print("****************************************************************")
        print("EXPERIMENT:", experiment.upper())
        print("****************************************************************\n")

        # execution_results = {}
        experiment_stats = pd.DataFrame()

        # RESET EXPERIMENT CONFIGS TO DEFAULT VALUES SINCE WE ARE USING GLOBAL VARIABLES...
        args.base_seed = 1230
        args.number_of_model_classes = None
        args.regularization_type = None
        args.regularization_value = 0

        experiment_configs = experiment.split("+")
        for config in experiment_configs:
            config = config.split("~")
            if config[0] == "bsd":
                args.base_seed = int(config[1])
                print("BASE SEED:", args.base_seed)
            elif config[0] == "nmc":
                args.number_of_model_classes = int(config[1])
                print("NUMBER OF MODEL CLASSES:", args.number_of_model_classes)
            elif config[0] == "rt":
                args.regularization_type = str(config[1])
                print("REGULARIZATION TYPE:", args.regularization_type)
            elif config[0] == "rv":
                args.regularization_value = float(config[1])
                print("REGULARIZATION VALUE:", args.regularization_value)

        args.experiment_path = os.path.join("artifacts", args.dataset, args.arch, experiment)
        print("PATH:", args.experiment_path)

        # for execution in range(1, args.executions + 1):
        for args.execution in range(1, args.executions + 1):

            # args.execution = execution
            execution_results = {}

            print("\n################ EXECUTION:", args.execution, "OF", args.executions, "################")

            # execute and get results and statistics...
            (execution_results["TRAIN [ACC1]"], execution_results["VAL [ACC1]"],
             execution_results["TRAIN LOSS"], execution_results["TRAIN ENTROPY"], execution_results["VAL ENTROPY"],
             execution_results["CPU TIME [MS]"], execution_results["GPU TIME [MS]"]) = execute()

            # appending results...
            experiment_stats = experiment_stats.append(execution_results, ignore_index=True)
            experiment_stats = experiment_stats[["TRAIN [ACC1]", "VAL [ACC1]", "TRAIN LOSS", "TRAIN ENTROPY",
                                                 "VAL ENTROPY", "CPU TIME [MS]", "GPU TIME [MS]"]]

        # print("\n################################\n", "EXPERIMENT STATISTICS", "\n################################\n")
        # print("\n", experiment.upper())
        # print("\n", experiment_stats.transpose())
        # print("\n", experiment_stats.describe())

        # Saving tab separeted files...
        csv.register_dialect('unixpwd', delimiter='\t', quoting=csv.QUOTE_NONE)
        for key in raw_results:
            file_path = os.path.join(args.experiment_path, key + '.data')
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = [item for item in range(1, args.executions+1)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='unixpwd')
                writer.writeheader()
                for epoch in range(len(raw_results[key])):
                    writer.writerow(raw_results[key][epoch])

        overall_stats[experiment] = experiment_stats

    print("\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", "OVERALL STATISTICS", "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    for key in overall_stats:
        print("\n", key.upper())
        print("\n", overall_stats[key].transpose())
        print("\n", overall_stats[key].describe())
        # print("\n", overall_stats[key].describe().loc[['mean', 'std']])

    print("\n\n\n")


if __name__ == '__main__':
    main()
