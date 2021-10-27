import builtins
from collections import Counter, OrderedDict
from random import shuffle
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import faiss

from custom_datasets import *


parser = argparse.ArgumentParser(description='NN evaluation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='name of the dataset.')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', type=str, default='alexnet',
                        choices=['alexnet' , 'resnet18' , 'resnet50', 'mobilenet' ,
                                 'moco_alexnet' , 'moco_resnet18' , 'moco_resnet50', 'moco_mobilenet', 'resnet50w5',
                                 'sup_alexnet' , 'sup_resnet18' , 'sup_resnet50', 'sup_mobilenet', 'pt_alexnet'])


parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='./output', type=str,
                    help='experiment output directory')
parser.add_argument('--indices', default='./indices', type=str,
                    help='experiment input directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('-k', default=1, type=int, help='k in kNN')
parser.add_argument('--splits', type=str, default='',
                    help='splits of unlabeled data to be labeled')
parser.add_argument('--name', default='', type=str,
                    help='method name to load indices from.')
parser.add_argument('--backbone', type=str, default='compress', 
                    help='name of method to do linear evaluation on.')

def main():
    global logger

    args = parser.parse_args()

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_val = 50000
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846
        args.num_val = 50000
        args.num_classes = 1000

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_val = 10000
        args.num_classes = 100

    elif args.dataset == "cifar10":
        args.num_images = 50000
        args.num_val = 10000
        args.num_classes = 10

    else:
        raise NotImplementedError

    if not os.path.exists(args.save):
        os.makedirs(args.save)


    main_worker(args)


def get_model(args):

    if args.arch == 'resnet18' :
        print(f"=> loading {args.backbone} weights ")
        model = models.__dict__[args.arch]()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'], strict=False)

        for param in model.parameters():
            param.requires_grad = False

        print(f"Weights of {args.backbone} loaded.")

    if args.arch == 'resnet50':
        print(f"=> loading {args.backbone} weights ")
        model = models.__dict__[args.arch]()
        checkpoint = torch.load(args.weights, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model.fc = nn.Sequential()

        for param in model.parameters():
            param.requires_grad = False

        model = torch.nn.DataParallel(model).cuda()
        print("=> loaded pre-trained model '{}'".format(args.weights))

    return model

def get_inference_loader(dataset, all_training_indices, args):
    if dataset == "imagenet":
        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        inference_loader = DataLoader(
            ImageNetSubset(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), all_training_indices),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "imagenet_lt":
        in_lt_train_txt = './data/ImageNet_LT_train.txt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        inference_loader = DataLoader(
            LT_Dataset(args.data, in_lt_train_txt, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])
        inference_loader = DataLoader(
            CIFAR100Subset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), all_training_indices),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
        inference_loader = DataLoader(
            CIFAR10Subset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), all_training_indices),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    else:
        raise NotImplementedError

    return inference_loader

def get_val_loader(dataset, all_val_indices, args):
    if dataset == "imagenet" or dataset == "imagenet_lt":
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_loader = DataLoader(
            ImageNetSubset(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), all_val_indices),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                    std=[0.2673, 0.2564, 0.2762])

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data, download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]), 
            train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar10":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data, download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])
            ]), 
            train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    else:
        raise NotImplementedError

    return val_loader


def main_worker(args):

    all_training_indices = np.arange(args.num_images)
    all_val_indices = np.arange(args.num_val)

    inference_loader = get_inference_loader(args.dataset, all_training_indices, args)
    val_loader = get_val_loader(args.dataset, all_val_indices, args)

    backbone = get_model(args)

    cudnn.benchmark = True

    inference_feats, inference_labels, val_feats, val_labels = get_feats(args.dataset, inference_loader, val_loader, backbone, args)

    splits = [int(x) for x in args.splits.split(',')]
    for split in splits:
        idx_file = '{}/{}_{}_{}.npy'.format(args.indices, args.name, args.dataset, split)
        current_indices = np.load(idx_file)
        print('current_indices loaded from {} with {} indices.'.format(idx_file, len(current_indices)))

        acc = faiss_knn(inference_feats[current_indices], inference_labels[current_indices], val_feats, val_labels, args.k)
        print(' * Acc for {} samples is {:.2f}'.format(split, acc))


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def faiss_knn(feats_train, targets_train, feats_val, targets_val, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    D, I = gpu_index.search(feats_val, k)

    pred = np.zeros(I.shape[0])
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i] = max(votes, key=lambda x: x[1])[0]

    acc = 100.0 * (pred == targets_val).mean()

    return acc


def get_feats(dataset, inference_loader, val_loader, backbone, args):
    if args.backbone == "compress":
        cached_inf_feats = '{}/inference_feats_{}_compress18_{}Teacher.pth.tar'.format(args.save, dataset, "MoCo" if "MoCo" in args.weights else "swAV")
        cached_val_feats = '{}/val_feats_{}_compress18_{}Teacher.pth.tar'.format(args.save, dataset, "MoCo" if "MoCo" in args.weights else "swAV")

    elif args.backbone == "moco":
        cached_inf_feats = '{}/inference_feats_{}_moco.pth.tar'.format(args.save, dataset)
        cached_val_feats = '{}/val_feats_{}_moco.pth.tar'.format(args.save, dataset)
    
    if args.load_cache and os.path.exists(cached_inf_feats):
        print(f'=> loading inference feats of {dataset} from cache: {cached_inf_feats}')
        inf_feats, inf_labels = torch.load(cached_inf_feats)

    else:
        print('get inference feats =>')

        backbone.eval()
        inf_feats, inf_labels, ptr = None, None, 0

        with torch.no_grad():
            for images, target, _ in tqdm(inference_loader):
                images = images.cuda(non_blocking=True)
                cur_targets = target.cpu()
                cur_feats = normalize(backbone(images)).cpu()
                B, D = cur_feats.shape
                inds = torch.arange(B) + ptr

                if not ptr:
                    inf_feats = torch.zeros((args.num_images, D)).float()
                    inf_labels = torch.zeros(args.num_images).long()

                inf_feats.index_copy_(0, inds, cur_feats)
                inf_labels.index_copy_(0, inds, cur_targets)
                ptr += B

        torch.save((inf_feats, inf_labels), cached_inf_feats)

    if args.load_cache and os.path.exists(cached_val_feats):
        print(f'=> loading val feats of {dataset} from cache: {cached_val_feats}')
        val_feats, val_labels = torch.load(cached_val_feats)

    else:
        print('get val feats =>')

        backbone.eval()
        val_feats, val_labels, ptr = None, None, 0

        with torch.no_grad():
            for images, target, _ in tqdm(val_loader):
                images = images.cuda(non_blocking=True)
                cur_targets = target.cpu()
                cur_feats = normalize(backbone(images)).cpu()
                B, D = cur_feats.shape
                inds = torch.arange(B) + ptr

                if not ptr:
                    val_feats = torch.zeros((args.num_val, D)).float()
                    val_labels = torch.zeros(args.num_val).long()

                val_feats.index_copy_(0, inds, cur_feats)
                val_labels.index_copy_(0, inds, cur_targets)
                ptr += B

        torch.save((val_feats, val_labels), cached_val_feats)

    return inf_feats, inf_labels, val_feats, val_labels

if __name__ == '__main__':
    main()

