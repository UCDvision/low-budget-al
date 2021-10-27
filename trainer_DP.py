#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
parser.add_argument('--lr_schedule', type=str, default='30,60,90',
                    help='lr drop schedule')
parser.add_argument('--splits', type=str, default='',
                    help='splits of unlabeled data to be labeled')
parser.add_argument('--name', type=str, default='',
                    help='name of method to do linear evaluation on.')

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

    else:
        raise NotImplementedError

    return train_loader

def get_val_loader(dataset, args):
    if dataset == "imagenet":
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

    else:
        raise NotImplementedError

    return val_loader

def main():

    args = parser.parse_args()

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100

    else:
        raise NotImplementedError


    print("=> creating model")
    task_model = models.__dict__[args.arch](num_classes=args.num_classes)
    task_model = torch.nn.DataParallel(task_model).cuda()

    optimizer = torch.optim.SGD(task_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            acc1 = checkpoint['best_acc1']
            task_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # validation data loading code
    val_loader = get_val_loader(args.dataset, args)

    splits = [int(x) for x in args.splits.split(',')]

    for split in splits:
        best_acc1 = 0

        cudnn.benchmark = True

        # current indices loading
        curr_idxs_file = '{}/{}_{}_{}.npy'.format(args.indices, args.name, args.dataset, split)
        if os.path.isfile(curr_idxs_file):
            print("=> Loading current indices: {}".format(curr_idxs_file))
            current_indices = np.load(curr_idxs_file)
            print('current indices size: {}.'.format(len(current_indices)))
        else:
            print("=> no such file found at '{}'".format(curr_idxs_file))

        # Training data loading code
        train_loader = get_train_loader(args.dataset, current_indices, args)

        print('Training task model started ...')
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, task_model, optimizer, epoch, args)

            # evaluate on validation set
            if epoch % 10 == 0 or epoch == args.epochs-1:
                acc1 = validate(task_model, val_loader, args)
                # remember best acc@1 and save checkpoint
                best_acc1 = max(acc1, best_acc1)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': task_model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, '{}_{}_{}.pth.tar'.format(args.name, args.dataset, split))
            
            lr_scheduler.step()
            print('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))    

        print('Final accuracy of {} labeled data is: {:.2f}'.format(split, acc1))

        print("=> creating model from scratch for the new split ...")
        task_model = models.__dict__[args.arch](num_classes=args.num_classes)
        task_model = torch.nn.DataParallel(task_model).cuda()

        args.start_epoch = 0
        optimizer = torch.optim.SGD(task_model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sched
        )

def train(train_loader, task_model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    task_model.train()

    end = time.time()

    for i, (images, target, _) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = task_model(images)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(task_model, val_loader, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    task_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = task_model(images)
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

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

    
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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


if __name__ == '__main__':
    main()
