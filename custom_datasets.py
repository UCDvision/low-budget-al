from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import numpy as np
from PIL import ImageFilter, Image
from tqdm import tqdm
import pandas as pd
import random
from typing import Callable, Optional
import os


class ImageNetSubset(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            indices = None
    ):
        super(ImageNetSubset, self).__init__(root, transform=transform)
        self.indices = indices

    def __getitem__(self, index):
        path, target = self.samples[self.indices[index]]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, target, self.indices[index]
    def __len__(self):
        return len(self.indices)

class CIFAR100Subset(Dataset):
    def __init__(self, path, transform, indices):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=transform)
        self.indices = indices

    def __getitem__(self, index):
        data, target = self.cifar100[self.indices[index]]        
        return data, target, self.indices[index]

    def __len__(self):
        return len(self.indices)

class CIFAR10Subset(Dataset):
    def __init__(self, path, transform, indices):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=transform)
        self.indices = indices

    def __getitem__(self, index):
        data, target = self.cifar10[self.indices[index]]        
        return data, target, self.indices[index]

    def __len__(self):
        return len(self.indices)

class LT_Dataset(Dataset):    
    def __init__(self, root, txt, transform=None, indices=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.indices = indices
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        if self.indices is not None:
            self.img_path = [self.img_path[i] for i in self.indices]
            self.labels = [self.labels[i] for i in self.indices]
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class ImageFolderEx(datasets.ImageFolder) :
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target