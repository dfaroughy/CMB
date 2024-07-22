import torch
import numpy as np
from dataclasses import dataclass
from cmb.data.utils import AbstractDataClass
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

import torch
import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from dataclasses import dataclass

class CIFARDataClass:
    def __init__(self, config: dataclass):
        self.config = config
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cifar_train = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        cifar_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

        if self.config.DATA_TARGET == 'cifar10':
            train_data = np.array(cifar_train.data)
            test_data = np.array(cifar_test.data)
            data = np.concatenate((train_data, test_data), axis=0)
            self.target = data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
            self.target = torch.tensor(self.target)

        if self.config.DATA_SOURCE == 'noise':
            self.source = torch.randn_like(self.target)

        train_targets = np.array(cifar_train.targets)
        test_targets = np.array(cifar_test.targets)
        self.context = np.concatenate((train_targets, test_targets), axis=0)
        self.context = torch.tensor(self.context)

        self.mask = torch.ones_like(self.target)

