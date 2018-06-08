# insert this to the top of your scripts (usually main.py)
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#    traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

import os
import argparse
import numpy
import random
import torch
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as torchvision_models
import models

from sklearn.metrics import accuracy_score


cudnn.benchmark = False
cudnn.deterministic = True

softmax = nn.Softmax(dim=1)

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

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('-dir', '--artifact-dir', type=str, metavar='DIR',
                    help='the project directory')
parser.add_argument('-data', '--dataset-dir', type=str, metavar='DATA',
                    help='output dir for logits extracted')
parser.add_argument('-lm', '--local-model', metavar='MODEL', default=None, choices=local_model_names,
                    help='model to be used: ' + ' | '.join(local_model_names))
parser.add_argument('-rm', '--remote-model', metavar='MODEL', default=None, choices=remote_model_names,
                    help='model to be used: ' + ' | '.join(remote_model_names))
parser.add_argument('-x', '--executions', default=5, type=int, metavar='N',
                    help='Number of executions (default: 5)')
parser.add_argument('-exps', '--experiments', default="baseline", type=str, metavar='EXPERIMENTS',
                    help='Experiments to be performed')
parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')
parser.add_argument('-gpu', '--gpu-id', default='1', type=str,
                    help='id for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
args.experiments = args.experiments.split("_")

# cuda device to use...
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


def execute():

    ######################################
    # Initial configuration...
    ######################################

    # Using seeds...
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Configuring args and dataset...
    if args.dataset == "mnist":
        args.number_of_dataset_classes = 10
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
        args.alpha_rank = args.number_of_model_classes
    elif args.dataset == "cifar10":
        args.number_of_dataset_classes = 10
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
        args.alpha_rank = args.number_of_model_classes
    elif args.dataset == "cifar100":
        args.number_of_dataset_classes = 100
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 100
        args.alpha_rank = 10
    else:
        args.number_of_dataset_classes = 1000
        args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 1000
        args.alpha_rank = 10

    # Defining the same normal classes for all experiments...
    args.normal_classes = sorted(random.sample(range(0, args.number_of_dataset_classes), args.number_of_model_classes))

    # Preparing paths...
    args.execution_path = os.path.join(args.experiment_path, "exec" + str(args.execution))
    # if not os.path.exists(args.execution_path):
    #     os.makedirs(args.execution_path)

    # Printing args...
    print("\nARGUMENTS: ", vars(args))

    # Preparing logger...
    # writer = SummaryWriter(log_dir=args.execution_path)
    # writer.add_text(str(vars(args)), str(vars(args)))

    ################################
    # Loading and prepraring data...
    ################################

    # basic load block to get logits, probabilits and targets tensors for train set...
    train_file = '{}/{}/{}.pth'.format(args.execution_path, 'logits', 'best_model_train')
    train_set = torch.load(train_file)
    train_logits = train_set[0]
    # train_logits_numpy = train_logits.numpy()
    train_probabilities = softmax(train_logits).detach()
    original_train_targets = train_set[1].long()
    closed_set_train_targets = [args.normal_classes.index(item) for item in original_train_targets]
    closed_set_train_targets_numpy = numpy.asarray(closed_set_train_targets)
    # open_set_train_targets = [int(item in args.normal_classes) for item in original_train_targets]
    # open_set_train_targets_numpy = numpy.asarray(open_set_train_targets)

    # basic load block to get logits, probabilits and targets tensors for val set...
    val_file = '{}/{}/{}.pth'.format(args.execution_path, 'logits', 'best_model_val')
    val_set = torch.load(val_file)
    val_logits = val_set[0]
    # val_logits_numpy = val_logits.numpy()
    val_probabilities = softmax(val_logits).detach()
    original_val_targets = val_set[1].long()
    closed_set_val_targets = [args.normal_classes.index(item) for item in original_val_targets]
    closed_set_val_targets_numpy = numpy.asarray(closed_set_val_targets)
    # open_set_val_targets = [int(item in args.normal_classes) for item in original_val_targets]
    # open_set_val_targets_numpy = numpy.asarray(open_set_val_targets)  # .numpy()

    # basic load block to get logits, probabilits and targets tensors for test set...
    test_file = '{}/{}/{}.pth'.format(args.execution_path, 'logits', 'best_model_test')
    test_set = torch.load(test_file)
    test_logits = test_set[0]
    # test_logits_numpy = test_logits.numpy()
    test_probabilities = softmax(test_logits).detach()
    original_test_targets = test_set[1].long()
    closed_set_test_targets = [args.normal_classes.index(item) for item in original_test_targets]
    closed_set_test_targets_numpy = numpy.asarray(closed_set_test_targets)
    # open_set_test_targets = [int(item in args.normal_classes) for item in original_test_targets]
    # open_set_test_targets_numpy = numpy.asarray(open_set_test_targets)  # .numpy()

    ################################
    # Training and testing...
    ################################

    # Calculating map acc1...
    map_train_predictions = train_probabilities.max(1)[1]  # get the index of the max probability
    map_train_predictions_numpy = map_train_predictions.numpy()
    map_train_acc1 = accuracy_score(closed_set_train_targets_numpy, map_train_predictions_numpy)
    map_val_predictions = val_probabilities.max(1)[1]  # get the index of the max probability
    map_val_predictions_numpy = map_val_predictions.numpy()
    map_val_acc1 = accuracy_score(closed_set_val_targets_numpy, map_val_predictions_numpy)
    map_test_predictions = test_probabilities.max(1)[1]  # get the index of the max probability
    map_test_predictions_numpy = map_test_predictions.numpy()
    map_test_acc1 = accuracy_score(closed_set_test_targets_numpy, map_test_predictions_numpy)

    # Calculating entropy...
    train_entropy = compute_entropy(train_logits, dim=1)
    train_mean_entropy = train_entropy.sum()/train_entropy.size(0)
    val_entropy = compute_entropy(val_logits, dim=1)
    val_mean_entropy = val_entropy.sum()/val_entropy.size(0)
    test_entropy = compute_entropy(test_logits, dim=1)
    test_mean_entropy = test_entropy.sum()/test_entropy.size(0)

    return (map_train_acc1, map_val_acc1, map_test_acc1,
            train_mean_entropy.item(), val_mean_entropy.item(), test_mean_entropy.item())


def compute_entropy(tensor, dim=1):
    local_softmax = nn.Softmax(dim=dim)
    logsoftmax = nn.LogSoftmax(dim=dim)
    softmax_output = local_softmax(tensor)
    logsoftmax_output = logsoftmax(tensor)
    return -(softmax_output * logsoftmax_output).sum(dim=dim)


def main():

    overall_stats = {}

    for experiment in args.experiments:

        print("\n\n")
        print("****************************************************************")
        print("EXPERIMENT:", experiment.upper())
        print("****************************************************************")

        # execution_results = {}
        experiment_stats = pd.DataFrame()

        if args.local_model is not None:
            args.arch = args.local_model
        else:
            args.arch = args.remote_model

        args.experiment_path = os.path.join("artifacts", args.dataset, args.arch, experiment)
        print("\nEXPERIMENT PATH:", args.experiment_path)

        # Configuring the args variables...
        args.number_of_model_classes = None
        # args.regularization_type = None
        # args.regularization_value = 0

        experiment_configs = experiment.split("+")
        for config in experiment_configs:
            config = config.split("~")
            if config[0] == "nmc":
                args.number_of_model_classes = int(config[1])
                print("NUMBER OF MODEL CLASSES:", args.number_of_model_classes)
            # elif config[0] == "rt":
            #     args.regularization_type = str(config[1])
            #     print("REGULARIZATION TYPE:", args.regularization_type)
            # elif config[0] == "rv":
            #     args.regularization_value = float(config[1])
            #     print("REGULARIZATION VALUE:", args.regularization_value)

        for args.execution in range(1, args.executions + 1):

            execution_results = {}

            print("\n################ EXECUTION:", args.execution, "of", args.executions, "################")

            # results and statistics...
            (execution_results["TRAIN ACC1"], execution_results["VAL ACC1"], execution_results["TEST ACC1"],
             execution_results["TRAIN ENTROPY"], execution_results["VAL ENTROPY"], execution_results["TEST ENTROPY"])\
                = execute()

            # appending results...
            experiment_stats = experiment_stats.append(execution_results, ignore_index=True)
            experiment_stats = experiment_stats[["TRAIN ACC1", "VAL ACC1", "TEST ACC1",
                                                "TRAIN ENTROPY", "VAL ENTROPY", "TEST ENTROPY"]]

        # print("\n################################\n", "EXPERIMENT STATISTICS", "\n################################\n")
        # print("\n", experiment.upper())
        # print("\n", experiment_stats.transpose())
        # print("\n", experiment_stats.describe())

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
