import torch
import numpy as np
from dataclasses import dataclass
from cmb.data.utils import AbstractDataClass
from torchvision import datasets, transforms

class CIFARDataClass(AbstractDataClass):
    def __init__(self, config: dataclass):

        self.config = config
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                            )

        cifar = datasets.CIFAR10(root='../data', 
                                 train=True, 
                                 download=False, 
                                 transform=transform)

        if self.config.DATA_TARGET == 'cifar10':
            self.target = cifar.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
            self.target = torch.tensor(self.target)

        if self.config.DATA_SOURCE == 'noise':
            self.source = torch.randn_like(self.target)

        self.context = cifar.targets 
        self.mask = torch.ones_like(self.target)