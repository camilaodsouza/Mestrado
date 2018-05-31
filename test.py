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
import scipy as sp
import sys
import math
# import seaborn as sns
# import torch.distributions.categorical.Categorical as Categorical
# import torch.distributions.kl.kl_divergence as kl_divergence
# import scipy.spatial.distance as spd

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import IsolationForest
from sklearn import svm

import models

try:
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()

WEIBULL_TAIL_SIZE = 20
# THRESHOLD = 0.01

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
parser.add_argument('-sd', '--seed', default='1230', type=int,
                    help='seed to be globaly used')
# parser.add_argument('-t', '--threshold', default=0.01, type=float, metavar='T',
#                    help='Threshold to be used')

args = parser.parse_args()
args.experiments = args.experiments.split("_")

# cuda device to use...
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


def execute():

    ######################################
    # Initial configuration...
    ######################################

    # Using seeds...
    args.execution_seed = args.seed + args.execution
    random.seed(args.execution_seed)
    numpy.random.seed(args.execution_seed)
    torch.manual_seed(args.execution_seed)
    torch.cuda.manual_seed(args.execution_seed)
    print("EXECUTION SEED:", args.execution_seed)
    # random.seed(args.seed)
    # numpy.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

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
    train_logits_numpy = train_logits.numpy()
    train_probabilities = compute_probabilities(train_logits)
    original_train_targets = train_set[1].long()
    closed_set_train_targets = [args.normal_classes.index(item) for item in original_train_targets]
    closed_set_train_targets_numpy = numpy.asarray(closed_set_train_targets)
    # open_set_train_targets = [int(item in args.normal_classes) for item in original_train_targets]
    # open_set_train_targets_numpy = numpy.asarray(open_set_train_targets)

    # basic load block to get logits, probabilits and targets tensors for val set...
    val_file = '{}/{}/{}.pth'.format(args.execution_path, 'logits', 'best_model_val')
    val_set = torch.load(val_file)
    val_logits = val_set[0]
    val_logits_numpy = val_logits.numpy()
    val_probabilities = compute_probabilities(val_logits)
    original_val_targets = val_set[1].long()
    closed_set_val_targets = [args.normal_classes.index(item) for item in original_val_targets]
    closed_set_val_targets_numpy = numpy.asarray(closed_set_val_targets)
    # open_set_val_targets = [int(item in args.normal_classes) for item in original_val_targets]
    # open_set_val_targets_numpy = numpy.asarray(open_set_val_targets)  # .numpy()

    # basic load block to get logits, probabilits and targets tensors for test set...
    test_file = '{}/{}/{}.pth'.format(args.execution_path, 'logits', 'best_model_test')
    test_set = torch.load(test_file)
    test_logits = test_set[0]
    test_logits_numpy = test_logits.numpy()
    test_probabilities = compute_probabilities(test_logits)
    original_test_targets = test_set[1].long()
    # closed_set_test_targets = [args.normal_classes.index(item) for item in original_test_targets]
    # closed_set_test_targets_numpy = numpy.asarray(closed_set_test_targets)
    open_set_test_targets = [int(item in args.normal_classes) for item in original_test_targets]
    open_set_test_targets_numpy = numpy.asarray(open_set_test_targets)  # .numpy()

    ################################
    # Training and testing...
    ################################

    """
    entropic_means = meta_learning(val_probabilities, val_targets, args.number_of_classes)
    print(entropic_means)
    """

    # Calculating map val acc1...
    map_val_predictions = val_probabilities.max(1)[1]  # get the index of the max probability
    map_val_predictions_numpy = map_val_predictions.numpy()
    # print("PREDICTIONS:\t", map_val_predictions_numpy[:30])
    # print("TARGETS:\t", closed_set_val_targets_numpy[:30])
    map_val_acc1 = accuracy_score(closed_set_val_targets_numpy, map_val_predictions_numpy)

    # Calculating train and test mean train entropy...
    train_entropy = compute_entropies(train_logits, dim=1)
    # train_mean_entropy = train_entropy.data.sum()/train_entropy.data.size(0)
    train_mean_entropy = train_entropy.sum()/train_entropy.size(0)
    val_entropy = compute_entropies(val_logits, dim=1)
    # val_mean_entropy = val_entropy.data.sum()/val_entropy.data.size(0)
    val_mean_entropy = val_entropy.sum()/val_entropy.size(0)
    test_entropy = compute_entropies(test_logits, dim=1)
    # test_mean_entropy = test_entropy.data.sum()/test_entropy.data.size(0)
    test_mean_entropy = test_entropy.sum()/test_entropy.size(0)

    # Calculating threshold auc...
    threshold_test_probabilities = test_probabilities.max(1)[0]  # get the max probability
    threshold_test_probabilities_numpy = threshold_test_probabilities.numpy()
    # print(threshold_test_probabilities_numpy[:30])
    threshold_auc = roc_auc_score(open_set_test_targets_numpy, threshold_test_probabilities_numpy)

    # """
    # Calculating openmax auc...
    distance_type = "euclidean"
    logits_means = calculate_means(
        # train_logits, original_train_targets, args.number_of_model_classes, args.normal_classes)
        train_logits, closed_set_train_targets, args.number_of_model_classes, args.normal_classes)
    # print("\nLOGITS MEANS:\n", logits_means)
    logits_means_distances = calculate_distances(
        # train_logits, original_train_targets, args.number_of_model_classes, args.normal_classes,
        train_logits, closed_set_train_targets, args.number_of_model_classes, args.normal_classes,
        logits_means, distance_type)
    # print("\nLOGITS DISTANCES:\t", logits_means_distances)
    weibull_models = weibull_tail_fitting(logits_means, logits_means_distances, args.number_of_model_classes)
    # print("\nWEIBULL MODELS LEN:\t", len(weibull_models))
    openmax_test_probabilities = compute_openmax(
        test_logits.data.numpy(), weibull_models, distance_type)
    # print(openmax_test_probabilities)
    openmax_test_knowing_probabilities = openmax_test_probabilities[:, :5]
    openmax_test_max_knowing_probabilities = openmax_test_knowing_probabilities.max(1)[0]
    openmax_test_max_knowing_probabilities_numpy = openmax_test_max_knowing_probabilities.numpy()
    # print(openmax_test_max_knowing_probabilities_numpy[:30])
    openmax_auc = roc_auc_score(open_set_test_targets_numpy, openmax_test_max_knowing_probabilities_numpy)
    # openmax_auc = openmax_auc if openmax_auc > 0.5 else 1 - openmax_auc
    # """

    # Calculating isoforest auc...
    isolation_forest = IsolationForest(n_estimators=400, max_samples=1.0, contamination=0.0, n_jobs=-1)
    isolation_forest.fit(train_logits_numpy)
    isolation_forest_test_score = isolation_forest.decision_function(test_logits_numpy)
    isolation_forast_auc = roc_auc_score(open_set_test_targets_numpy, isolation_forest_test_score)
    # isolation_forast_auc = isolation_forast_auc if isolation_forast_auc > 0.5 else 1 - isolation_forast_auc
    # fpr, tpr, _ = roc_curve(open_set_test_targets_numpy, isolation_forest_test_score)
    """
    if args.dataset in {"mnist", "cifar10"}:
        isolation_forest = IsolationForest(n_estimators=400, max_samples=1.0, contamination=0.0, n_jobs=-1)
    else:
        isolation_forest = IsolationForest(contamination=0.0, n_jobs=-1)
    isolation_forest.fit(train_logits_numpy)
    if args.dataset in {"mnist", "cifar10", "cifar100"}:
        isolation_forest_test_score = isolation_forest.decision_function(test_logits_numpy)
    else:
        isolation_forest_test_score = -isolation_forest.decision_function(test_logits_numpy)
    isolation_forast_auc = roc_auc_score(open_set_test_targets_numpy, isolation_forest_test_score)
    fpr, tpr, _ = roc_curve(open_set_test_targets_numpy, isolation_forest_test_score)
    """

    """
    # Calculating ocsvm auc...
    oc_svm = svm.OneClassSVM()
    oc_svm.fit(train_logits_numpy)
    oc_svm_test_score = oc_svm.decision_function(test_logits_numpy)
    oc_svm_auc = roc_auc_score(open_set_test_targets_numpy, oc_svm_test_score)
    fpr, tpr, _ = roc_curve(open_set_test_targets_numpy, oc_svm_test_score)
    """

    return (100 * map_val_acc1, train_mean_entropy.item(), val_mean_entropy.item(), test_mean_entropy.item(),
            threshold_auc, openmax_auc, isolation_forast_auc, 0)


def weibull_tail_fitting(vectors_means, vectors_distances, number_of_model_classes, tailsize=WEIBULL_TAIL_SIZE):
    weibull_models = {}

    for item in range(number_of_model_classes):

        weibull_models[item] = {}
        weibull_models[item]["mean"] = vectors_means[item]
        weibull_models[item]["distances"] = vectors_distances[item]

        tailtofit = sorted(weibull_models[item]["distances"])[-tailsize:]
        mr = libmr.MR()
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_models[item]["model"] = mr

    return weibull_models


def compute_openmax(logits, weibull_models, distance_type):
    """
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained from weibull_tail_fitting() function
    labellist : ImageNet 2012 labellist
    imgarr : features for a particular image extracted using caffe architecture

    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using SoftMax (these
    were precomputed from caffe architecture. Function returns them for the sake
    of convienence)

    """

    # print(logits.shape)
    # logits = list(logits)
    # print(logits[0].shape)

    full_openmax_probab = []

    for logit in list(logits):

        # imglayer = imgarr[layer]
        # ranked_list = imgarr['scores'].argsort().ravel()[::-1]
        ranked_list = logit.argsort().ravel()[::-1]
        alpha_weights = [((args.alpha_rank + 1) - i) / float(args.alpha_rank) for i in range(1, args.alpha_rank + 1)]
        # ranked_alpha = sp.zeros(1000)
        ranked_alpha = sp.zeros(args.number_of_model_classes)

        # print(ranked_list.size)
        # print(len(alpha_weights))
        # print(ranked_alpha.size)

        for i in range(len(alpha_weights)):
            ranked_alpha[ranked_list[i]] = alpha_weights[i]

        # Now recalibrate each fc8 score for each channel and for each class
        # to include probability of unknown
        openmax_fc8, openmax_score_u = [], []

        # for channel in range(NCHANNELS):
        #     logit = imglayer[channel, :]

        openmax_logit = []
        openmax_unknown = []
        # count = 0

        # for channel in range(NCHANNELS):
        for categoryid in range(args.number_of_model_classes):
            # get distance_type between current channel and mean vector
            # category_weibull = query_weibull(labellist[categoryid], weibull_model, distance_type=distance_type)
            # channel_distance = compute_distance(
            #    channel_scores, channel, category_weibull[0], distance_type=distance_type)
            logit_distance = calculate_distance(logit, weibull_models[categoryid]["mean"], distance_type)

            # obtain w_score for the distance_type and compute probability of the distance_type
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            # wscore = category_weibull[2][channel].w_score(channel_distance)
            wscore = weibull_models[categoryid]["model"].w_score(logit_distance)
            # modified_fc8_score = channel_scores[categoryid] * (1 - wscore * ranked_alpha[categoryid])
            modified_logit = logit[categoryid] * (1 - wscore * ranked_alpha[categoryid])
            # openmax_fc8_channel += [modified_fc8_score]
            openmax_logit += [modified_logit]
            # openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score]
            openmax_unknown += [logit[categoryid] - modified_logit]

        # gather modified scores fc8 scores for each channel for the given image
        openmax_fc8 += [openmax_logit]
        openmax_score_u += [openmax_unknown]
        # End of original channel iteraction...

        openmax_fc8 = sp.asarray(openmax_fc8)
        openmax_score_u = sp.asarray(openmax_score_u)

        # Pass the recalibrated fc8 scores for the image into openmax
        openmax_probab = calculate_openmax_probabilities(openmax_fc8, openmax_score_u)
        # softmax_probab = imgarr['scores'].ravel()

        # Accumulationg openmax probabilities...
        full_openmax_probab += [openmax_probab]

        # return sp.asarray(openmax_probab)

    # return sp.asarray(full_openmax_probab)
    return torch.Tensor(full_openmax_probab)


def calculate_openmax_probabilities(openmax_fc8, openmax_score_u):
    """ Convert the scores in probability value using openmax

    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class

    """
    prob_scores, prob_unknowns = [], []

    # for channel in range(NCHANNELS):
    channel_scores, channel_unknowns = [], []

    for category in range(args.number_of_model_classes):
        # channel_scores += [sp.exp(openmax_fc8[channel, category])]
        channel_scores += [sp.exp(openmax_fc8[0, category])]

    # total_denominator = sp.sum(sp.exp(openmax_fc8[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))
    total_denominator = sp.sum(sp.exp(openmax_fc8[0, :])) + sp.exp(sp.sum(openmax_score_u[0, :]))
    prob_scores += [channel_scores / total_denominator]
    # prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :])) / total_denominator]
    prob_unknowns += [sp.exp(sp.sum(openmax_score_u[0, :])) / total_denominator]
    # End of original channel iteraction...

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis=0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    # assert len(modified_scores) == 1001
    assert len(modified_scores) == args.number_of_model_classes + 1

    return modified_scores


def calculate_means(vectors, targets, number_of_model_classes, normal_classes):
    """
    vectors can be logits or probabilities
    """
    number_of_vectors = [0] * number_of_model_classes  # [0 for item in range(number_of_dataset_classes)]
    vectors_means = torch.FloatTensor(number_of_model_classes, number_of_model_classes).fill_(0)

    for item in range(len(vectors)):
        """
        if targets[item] in normal_classes:
            vectors_means.index_add_(0, torch.LongTensor([normal_classes.index(targets[item])]),
            # vectors_means.index_add_(0, torch.LongTensor([targets[item]]),
                                     vectors[item].unsqueeze(0))
            number_of_vectors[normal_classes.index(targets[item])] += 1
            # number_of_vectors[targets[item]] += 1
        """

        # vectors_means.index_add_(0, torch.LongTensor([normal_classes.index(targets[item])]),
        vectors_means.index_add_(0, torch.LongTensor([targets[item]]),
                                 vectors[item].unsqueeze(0))
        # number_of_vectors[normal_classes.index(targets[item])] += 1
        number_of_vectors[targets[item]] += 1

    for item in range(len(number_of_vectors)):
        vectors_means[item].div_(number_of_vectors[item])

    return vectors_means


def calculate_distances(vectors, targets, number_of_model_classes, normal_classes, vectors_means, distance_type):
    """
    vectors can be logits or probabilities
    """
    vectors_means_distances = [[] for item in range(number_of_model_classes)]

    for item in range(len(vectors)):
        # vectors_means_distances[normal_classes.index(targets[item])].append(
        vectors_means_distances[targets[item]].append(
            # calculate_distance(vectors[item], vectors_means[normal_classes.index(targets[item])], distance_type))
            calculate_distance(vectors[item], vectors_means[targets[item]], distance_type))

    return vectors_means_distances


def calculate_distance(vector1, vector2, distance_type):
    """
    vectors can be logits or probabilities
    """
    if distance_type == "euclidean":
        # return ((vector1 - vector2) * (vector1 - vector2)).sum()
        return math.sqrt(((vector1 - vector2) * (vector1 - vector2)).sum())
    elif distance_type == "relative entropy":
        eps = float(numpy.finfo(numpy.float32).eps)
        return (vector1*torch.log(vector1/(vector2+eps))).sum()
    else:
        return None


def relative_entropy_kernel(tensor1, tensor2):
    print("VECTOR1 TYPE:\t", tensor1.shape)
    print("VECTOR2 TYPE:\t", tensor2.shape)
    tensor1 = torch.from_numpy(tensor1).cuda()
    tensor2 = torch.from_numpy(tensor2).cuda()
    # tensor1 = Variable(tensor1)
    # tensor2 = Variable(tensor2)
    # tensor1_probabilities = softmax(tensor1).data
    # tensor2_probabilities = softmax(tensor2).data
    tensor1_probabilities = compute_probabilities(tensor1)  # .detach()
    tensor2_probabilities = compute_probabilities(tensor2)  # .detach()

    distances = []

    for item1 in tensor1_probabilities:
        for item2 in tensor2_probabilities:
            if len(distances) % 1000000 == 0:
                print(len(distances))
            distances.append(calculate_distance(item1, item2, "relative entropy"))

    return numpy.ndarray(distances)


def meta_learning(probabilities, targets, number_of_classes):
    print(number_of_classes)
    entropic_counts = [0] * number_of_classes  # [0 for item in range(number_of_classes)]
    entropic_means = torch.FloatTensor(number_of_classes, number_of_classes).fill_(0)

    print("META LEARNING TRAINING SIZE:", len(probabilities))

    for index in range(len(probabilities)):
        # print("INDEX:", index)
        # print("TARGET OF INDEX:", [targets[index]])
        # print("######", torch.LongTensor([targets[index]]).size())
        # print("######", probabilities.data[index].unsqueeze(0).size())
        # print("######", entropic_means.size())
        entropic_means.index_add_(
            0, torch.LongTensor([targets[index]]), probabilities.data[index].unsqueeze(0))
        entropic_counts[targets[index]] += 1

    for index in range(len(entropic_counts)):
        entropic_means[index].div_(entropic_counts[index])

    print("\nENTROPIC COUNTS SUM:\t", sum(entropic_counts))
    print("\nENTROPIC MEANS SUM:\t", entropic_means.sum())
    # print(entropic_means)

    """
    entropic_distances_totals = [0] * number_of_classes
    for index in range(len(probabilities)):
        entropic_distances_totals[targets[index]] += get_kl_distance(
            probabilities.data[index], entropic_means[targets[index]], exclude=None)
    entropic_distances_means = [entropic_distances_totals[item]/entropic_counts[item]
                                for item in range(len(entropic_distances_totals))]
    print(min(entropic_distances_means))
    print(max(entropic_distances_means))
    print(auc_statistics.mean(entropic_distances_means))
    print(auc_statistics.stdev(entropic_distances_means))
    # sns.set()
    # data = numpy.asarray(entropic_distances_means)
    # sns.distplot(data, bins=50)
    # plt.savefig("entropic_distances_means")
    """
    return entropic_means


def compute_red_predictions(probabilities, entropic_representations_means, threshold):
    red_predictions = torch.LongTensor(len(probabilities.data))
    for index in range(len(probabilities)):
        candidate_predictions = compute_candidate_predictions(probabilities.data[index], threshold)
        # print(candidate_predictions)
        normalized_kl_distances = {}
        for candidate_prediction in candidate_predictions:
            normalized_kl_distances[candidate_prediction] = get_kl_distance(
                probabilities.data[index], entropic_representations_means[candidate_prediction],
                exclude=None)  # /entropic_distances_means[candidate_prediction]
        red_predictions[index] = min(normalized_kl_distances, key=normalized_kl_distances.get)
    return red_predictions


def compute_candidate_predictions(probabilities, threshold):
    result = []
    for index in range(len(probabilities)):
        if probabilities[index] >= threshold:
            result.append(index)
    if not result:
        result = compute_candidate_predictions(probabilities, threshold / 2)
    return result


def get_kl_distance(p_probabilities, q_probabilities, exclude=None):
    eps = float(numpy.finfo(numpy.float32).eps)
    # p = Categorical(p_probabilities)
    # q = Categorical(q_probabilities)
    # return float(kl_divergence(p, q))
    # temp = 0
    # if exclude is not None:
    #    temp = p_probabilities[exclude]
    #    p_probabilities[exclude] = eps
    # result = (p_probabilities*torch.log(p_probabilities/(q_probabilities+eps))).sum()
    # if exclude is not None:
    #    p_probabilities[exclude] = temp
    # return result
    exclude_term = 0
    if exclude is not None:
        exclude_term = (p_probabilities[exclude]*math.log(p_probabilities[exclude]/(q_probabilities[exclude]+eps)))
    full_kl_distance = (p_probabilities*torch.log(p_probabilities/(q_probabilities+eps))).sum()
    return full_kl_distance - exclude_term


def compute_probabilities(tensor, dim=1):
    return nn.Softmax(dim=dim)(tensor)


def compute_entropies(tensor, dim=1):
    softmax = nn.Softmax(dim=dim)
    logsoftmax = nn.LogSoftmax(dim=dim)
    softmax_output = softmax(tensor)  # .data)#.data
    logsoftmax_output = logsoftmax(tensor)  # .data)#.data
    return -(softmax_output * logsoftmax_output).sum(dim=dim)


def main():

    overall_stats = {}

    for experiment in args.experiments:

        print("\n\n")
        print("****************************************************************")
        print("EXPERIMENT:", experiment.upper())
        print("****************************************************************\n")

        # execution_results = {}
        experiment_stats = pd.DataFrame()

        if args.local_model is not None:
            args.arch = args.local_model
        else:
            args.arch = args.remote_model

        experiment_configs = experiment.split("+")
        for config in experiment_configs:
            config = config.split("~")
            if config[0] == "ebs":
                args.seed = int(config[1])
                print("EXPERIMENT BASE SEED:", args.seed)
            elif config[0] == "nmc":
                args.number_of_model_classes = int(config[1])
                print("NUMBER OF MODEL CLASSES:", args.number_of_model_classes)
            # elif config[0] == "rt":
            #     args.regularization_type = str(config[1])
            #     print("REGULARIZATION TYPE:", args.regularization_type)
            # elif config[0] == "rv":
            #     args.regularization_value = float(config[1])
            #     print("REGULARIZATION VALUE:", args.regularization_value)

        args.experiment_path = os.path.join("artifacts", args.dataset, args.arch, experiment)
        print("EXPERIMENT PATH:", args.experiment_path)

        for args.execution in range(1, args.executions + 1):

            # args.execution = execution
            execution_results = {}

            print("\n################ EXECUTION:", args.execution, "of", args.executions, "################")

            # execute and get results and statistics...
            (execution_results["MAP VAL [ACC1]"], execution_results["TRAIN ENTROPY"],
             execution_results["VAL ENTROPY"], execution_results["TEST ENTROPY"],
             execution_results["THRESHOLD [AUC]"], execution_results["OPENMAX [AUC]"],
             execution_results["ISOFOREST [AUC]"], execution_results["OCSVM [AUC]"]) = execute()

            # appending results...
            experiment_stats = experiment_stats.append(execution_results, ignore_index=True)
            experiment_stats = experiment_stats[["MAP VAL [ACC1]", "TRAIN ENTROPY", "VAL ENTROPY", "TEST ENTROPY",
                                                 "THRESHOLD [AUC]", "OPENMAX [AUC]", "ISOFOREST [AUC]", "OCSVM [AUC]"]]

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
        # print("\n", overall_stats[key].describe().loc[['mean','std']])

    print("\n\n\n")


if __name__ == '__main__':
    main()
