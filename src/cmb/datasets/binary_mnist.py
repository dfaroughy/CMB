import torch
import numpy as np
from dataclasses import dataclass
from torchvision import datasets, transforms


class CouplingData:
    ''' Dataclass for the coupling between target and source datasets.
    '''
    def __init__(self, config: dataclass):

        self.config = config

        #...2D Targets:

        if config.target == 'digits':  self.target = BinaryMNIST(labels_as_context=False, flatten_output=True)
        elif config.target == 'fashion':  self.target = BinaryFashionMNIST(labels_as_context=False, flatten_output=True)
        else: raise ValueError('Unknown target dataset.')

        #...2D Sources:
            
        if config.source == 'noise': self.source = BinaryNoiseImages(num_img=self.target.num_img, img_resolution=(28, 28), flatten_output=True)
        elif config.source == 'digits':  self.source = BinaryMNIST(labels_as_context=False, flatten_output=True)
        elif config.source == 'fashion':  self.source = BinaryFashionMNIST(labels_as_context=False, flatten_output=True)
        else: raise ValueError('Unknown source dataset.')

class BinaryMNIST:
    def __init__(self, threshold: float = 0.5, labels_as_context=False, flatten_output=False):
        self.threshold = threshold
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        mnist_train = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)
        train_data = np.array(mnist_train.data)
        test_data = np.array(mnist_test.data)
        data = np.concatenate((train_data, test_data), axis=0)

        self.discrete = (data > (self.threshold * 255)).astype(np.float32)
        self.discrete = self.discrete[:, None, :, :]  # Add channel dimension
        self.discrete = torch.tensor(self.discrete)

        self.num_img, _, w, h = self.discrete.shape

        if flatten_output:
            self.discrete = self.discrete.view(self.num_img, w*h)

        if labels_as_context:
            train_targets = np.array(mnist_train.targets)
            test_targets = np.array(mnist_test.targets)
            self.context = np.concatenate((train_targets, test_targets), axis=0)
            self.context = torch.tensor(self.context)
    
    def __len__(self):
        return self.num_img

class BinaryFashionMNIST:
    def __init__(self, threshold: float = 0.7, labels_as_context=False, flatten_output=False):
        self.threshold = threshold
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        mnist_train = datasets.FashionMNIST(root='../../data', train=True, download=True, transform=transform)
        mnist_test = datasets.FashionMNIST(root='../../data', train=False, download=True, transform=transform)
        train_data = np.array(mnist_train.data)
        test_data = np.array(mnist_test.data)
        data = np.concatenate((train_data, test_data), axis=0)

        self.discrete = (data > (self.threshold * 255)).astype(np.float32)
        self.discrete = self.discrete[:, None, :, :]  # Add channel dimension
        self.discrete = torch.tensor(self.discrete)

        self.num_img, _, w, h = self.discrete.shape

        if flatten_output:
            self.discrete = self.discrete.view(self.num_img, w*h)

        if labels_as_context:
            train_targets = np.array(mnist_train.targets)
            test_targets = np.array(mnist_test.targets)
            self.context = np.concatenate((train_targets, test_targets), axis=0)
            self.context = torch.tensor(self.context)
    
    def __len__(self):
        return self.num_img


class BinaryNoiseImages:
    def __init__(self, num_img=70000, img_resolution=(28, 28), threshold: float = 0.5, flatten_output=False):
        self.discrete = torch.bernoulli(torch.full((num_img, 1, *img_resolution), threshold))
        n, _, w, h = self.discrete.shape
        if flatten_output:
            self.discrete = self.discrete.view(n, w*h)
    
    def __len__(self):
        return self.num_img