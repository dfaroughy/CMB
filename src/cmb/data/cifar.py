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

        # Combine the targets from both datasets
        train_targets = np.array(cifar_train.targets)
        test_targets = np.array(cifar_test.targets)
        self.context = np.concatenate((train_targets, test_targets), axis=0)
        self.context = torch.tensor(self.context)

        self.mask = torch.ones_like(self.target)

# Example config dataclass
@dataclass
class Configs:
    DATA_TARGET: str
    DATA_SOURCE: str

conf = Configs(DATA_TARGET='cifar10', DATA_SOURCE='noise')
cifar = CIFARDataClass(conf)


# class CIFARDataClass(AbstractDataClass):
#     def __init__(self, config: dataclass):

#         self.config = config
#         transform = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#                                             )

#         cifar_train = datasets.CIFAR10(root='../data',  train=True,  download=True, transform=transform)
#         cifar_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
#         cifar = ConcatDataset([cifar_train, cifar_test])

#         if self.config.DATA_TARGET == 'cifar10':
#             self.target = cifar.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
#             self.target = torch.tensor(self.target)

#         if self.config.DATA_SOURCE == 'noise':
#             self.source = torch.randn_like(self.target)

#         self.context = cifar.targets 
#         self.mask = torch.ones_like(self.target)