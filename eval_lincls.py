import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
from typing import Callable, Optional
import faiss

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

import strategies
from custom_datasets import *


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='name of the dataset.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet50',
                    help='model architecture')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output', type=str,
                    help='experiment output directory')
parser.add_argument('--indices', default='./indices', type=str,
                    help='experiment input directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', default='', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--lr_schedule', type=str, default='30,60,90',
                    help='lr drop schedule')
parser.add_argument('--splits', type=str, default='',
                    help='splits of unlabeled data to be labeled')
parser.add_argument('--name', type=str, default='',
                    help='name of method to do linear evaluation on.')
parser.add_argument('--backbone', type=str, default='compress', 
                    help='name of method to do linear evaluation on.')


def main():

    args = parser.parse_args()

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846
        args.num_classes = 1000

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100

    elif args.dataset == "cifar10":
        args.num_images = 50000
        args.num_classes = 10

    else:
        raise NotImplementedError

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    main_worker(args)


def load_weights(model, wts_path, args):
    if args.backbone == "compress":
        # each pre-trained model has a different output size
        # it's 128 for MoCoTeacher
        # and 2048 for swAVTeacher
        model.fc = nn.Linear(model.fc.weight.shape[1], 128 if "MoCo" in wts_path else 2048)
        if os.path.exists(wts_path):
            print(f"=> loading weights from {args.backbone}")
            wts = torch.load(wts_path)
            if 'state_dict' in wts:
                ckpt = wts['state_dict']
            if 'model' in wts:
                ckpt = wts['model']
            else:
                ckpt = wts

            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            state_dict = {}

            for m_key, m_val in model.state_dict().items():
                if m_key in ckpt:
                    state_dict[m_key] = ckpt[m_key]
                else:
                    state_dict[m_key] = m_val
                    print('not copied => ' + m_key)

            model.load_state_dict(state_dict)
            
            print(f"Weights of {args.backbone} loaded.")
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(wts_path))

    elif args.backbone == "moco":
        if os.path.isfile(wts_path):
            print("=> loading checkpoint '{}'".format(wts_path))
            checkpoint = torch.load(wts_path, map_location="cpu")

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
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, msg.missing_keys

            print("=> loaded pre-trained model '{}'".format(wts_path))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(wts_path))

def get_model(arch, wts_path, args):
    model = models.__dict__[arch]()
    load_weights(model, wts_path, args)
    model.fc = nn.Sequential()

    for p in model.parameters():
        p.requires_grad = False

    return model


def get_inference_loader(dataset, all_indices, args):
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
            ]), all_indices),
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
            ]), all_indices),
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
            ]), all_indices),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    else:
        raise NotImplementedError

    return inference_loader

def get_train_loader(dataset, current_indices, args):
    if dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_loader = DataLoader(
            ImageNetSubset(os.path.join(args.data, 'train'), 
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "imagenet_lt":
        in_lt_train_txt = './data/ImageNet_LT_train.txt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = DataLoader(
            LT_Dataset(args.data, in_lt_train_txt, 
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])
        train_loader = DataLoader(
            CIFAR100Subset(args.data, transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
        train_loader = DataLoader(
            CIFAR10Subset(args.data, transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )


    else:
        raise NotImplementedError

    return train_loader

def get_val_loader(dataset, args):
    if dataset == "imagenet" or dataset == "imagenet_lt":
        valdir = os.path.join(args.data, 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data, download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                    std=[0.2673, 0.2564, 0.2762])
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
    all_indices = np.arange(args.num_images)

    inference_loader = get_inference_loader(args.dataset, all_indices, args)
    backbone = get_model(args.arch, args.weights, args)
    backbone = nn.DataParallel(backbone).cuda()

    cudnn.benchmark = True

    # get all dataset labels in eval mode
    _, inference_labels = get_feats(args.dataset, inference_loader, backbone, args)

    # validation data loading code
    val_loader = get_val_loader(args.dataset, args)

    splits = [int(x) for x in args.splits.split(',')]
    for split in splits:

        # current indices loading
        curr_idxs_file = '{}/{}_{}_{}.npy'.format(args.indices, args.name, args.dataset, split)
        if os.path.isfile(curr_idxs_file):
            print("=> Loading current indices: {}".format(curr_idxs_file))
            current_indices = np.load(curr_idxs_file)
            print('current indices size: {}. {}% of all categories is seen'.format(len(current_indices), len(np.unique(inference_labels[current_indices])) / args.num_classes * 100))
        else:
            raise ValueError("=> no index file found at '{}'".format(curr_idxs_file))

        # Training data loading code
        train_loader = get_train_loader(args.dataset, current_indices, args)

        print("=> creating model")
        linear = nn.Sequential(
            Normalize(),
            nn.Linear(get_channels(args.arch), args.num_classes),
        )
        linear = linear.cuda()

        # optimizer = torch.optim.SGD(linear.parameters(),
        #                             args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(linear.parameters(),
                                        args.lr
                                        )
        sched = [int(x) for x in args.lr_schedule.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sched
        )

        cudnn.benchmark = True
        print('Training task model started ...')
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, backbone, linear, optimizer, epoch, args)

            # evaluate on validation set
            if epoch % args.print_freq == 0 or epoch == args.epochs-1:
                acc1 = validate(val_loader, backbone, linear, args)

            # modify lr
            lr_scheduler.step()
            print('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))
            
        print('Final accuracy of {} unlabeled data is: {}'.format(split, acc1))


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'resnet50':
        c = 2048
    elif arch == 'resnet18':
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    elif arch == 'resnet50x5_swav':
        c = 10240
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def get_feats(dataset, loader, model, args):
    if args.backbone == "compress":
        cached_feats = '{}/inference_feats_{}_compress18_swAVTeacher.pth.tar'.format(args.save, dataset)
    elif args.backbone == "moco":
        cached_feats = '{}/inference_feats_{}_moco.pth.tar'.format(args.save, dataset)

    if args.load_cache and os.path.exists(cached_feats):
        print(f'=> loading inference feats of {dataset} from cache: {cached_feats}')
        return torch.load(cached_feats)
    else:
        print('get inference feats =>')

        model.eval()
        feats, labels, ptr = None, None, 0

        with torch.no_grad():
            for images, target, _ in tqdm(loader):
                images = images.cuda(non_blocking=True)
                cur_targets = target.cpu()
                cur_feats = normalize(model(images)).cpu()
                B, D = cur_feats.shape
                inds = torch.arange(B) + ptr

                if not ptr:
                    feats = torch.zeros((len(loader.dataset), D)).float()
                    labels = torch.zeros(len(loader.dataset)).long()

                feats.index_copy_(0, inds, cur_feats)
                labels.index_copy_(0, inds, cur_targets)
                ptr += B

        torch.save((feats, labels), cached_feats)
        return feats, labels

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()
