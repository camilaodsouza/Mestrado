# insert this to the top of your scripts (usually main.py)
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#    traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

import argparse
import fnmatch
import io
import os

# from utils import update_progress  # , _order_files

dataset_names = ('cifar10', 'cifar100', 'mnist')

parser = argparse.ArgumentParser(description='Crate Images Manifest')

parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')

args = parser.parse_args()


def update_progress(progress):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50), progress * 100), end="")


def create_manifest(dataset, relative_data_path, tag):
    # manifest_path = tag + '_manifest.csv'
    # data_path = dataset + relative_data_path
    data_path = os.path.join(dataset, relative_data_path)
    # print(dataset)
    # print(data_path)
    manifest_path = os.path.join(dataset, tag + '_manifest.csv')
    file_paths = []
    img_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(data_path)
                 for f in fnmatch.filter(files, '*.jpg')]
    img_files += [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.bmp')]
    size = len(img_files)
    counter = 0
    for file_path in img_files:
        file_paths.append(file_path.strip())
        counter += 1
        update_progress(counter / float(size))
    print('\n')
    counter = 0
    with io.FileIO(manifest_path, "w") as file:
        for img_path in file_paths:
            sample = os.path.abspath(img_path)
            # print(sample.replace(os.path.abspath(data_path), ''))
            # label = str(sample.split('/')[10].lstrip('class'))
            # id = str(sample.split('/')[11].lstrip('image').rstrip('.jpg'))
            aux = str(sample.replace(os.path.abspath(data_path), ''))
            label = aux.replace('/class', '').split('/')[0]
            id = aux.replace('/class', '').split('/')[1].replace('image', '').replace('.jpg', '')
            example = sample + ',' + label + ',' + id + '\n'
            # print(sample.split('/')[10].strip('class'))
            file.write(example.encode('utf-8'))
            counter += 1
            update_progress(counter / float(size))
    print('\n')


def main():
    # train_path = args.dataset + '/images/train/'
    # test_path = args.dataset + '/images/val/'
    print('\n', 'Creating manifests...')
#    create_manifest(train_path, args.dataset + '_train')
#    create_manifest(test_path, args.dataset + '_val')
    create_manifest(args.dataset, 'images/train/',  'train')
    create_manifest(args.dataset, 'images/val/', 'val')


if __name__ == '__main__':
    main()
