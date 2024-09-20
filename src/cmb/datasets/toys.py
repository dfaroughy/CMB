import torch
import numpy as np
from dataclasses import dataclass
import seaborn as sns
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt
import math

@dataclass
class Configs:
    target : str = 'Ngaussians'
    source : str = 'StdGauss'
    num_points : int = 8000
    dim_continuous : int = 2
    dim_discrete : int = 0
    dim_context : int = 0
    num_gaussians : int = 8

class CouplingData:
    ''' Dataclass for the coupling between target and source datasets.
    '''
    def __init__(self, config: Configs):

        self.config = config

        #...2D Targets:

        if config.target == 'NGaussians': 
            self.target = NGaussians(num_gaussians=config.num_gaussians, 
                                    num_colors=config.vocab_size,
                                    num_points_per_gaussian=config.num_points//config.num_gaussians , 
                                    std_dev=0.1, 
                                    scale=5, 
                                    labels_as_state=bool(config.dim.discrete),
                                    labels_as_context=bool(config.dim.context))
            
        elif config.target == 'TwoMoons': 
            self.target = TwoMoons(num_points_per_moon=config.num_points//2, 
                                    std_dev=0.2, 
                                    labels_as_state=bool(config.dim.discrete),
                                    labels_as_context=bool(config.dim.context))
        else:
            raise ValueError('Unknown target dataset.')

        #...2D Sources:
            
        if config.source == 'NGaussians':
            self.source = NGaussians(num_gaussians=config.num_gaussians, 
                                     num_colors=config.vocab_size,
                                     num_points_per_gaussian=config.num_points//config.num_gaussians , 
                                     std_dev=0.1, 
                                     scale=5, 
                                     labels_as_state=bool(config.dim_discrete),
                                     labels_as_context=bool(config.dim_context))
            
        elif config.source == 'TwoMoons':
            self.source = TwoMoons(num_points_per_moon=config.num_points//2, 
                                   std_dev=0.2, 
                                   labels_as_state=bool(config.dim_discrete),
                                   labels_as_context=bool(config.dim_context))
            
        elif config.source == 'StdGauss':
            self.source = StdGauss(num_colors=config.vocab_size,
                                   num_points=config.num_points, 
                                   std_dev=0.5, 
                                   labels_as_state=bool(config.dim_discrete),
                                   labels_as_context=bool(config.dim_context))
        else:
            raise ValueError('Unknown source dataset.')
        

class NGaussians:
    def __init__(self, dim=2, num_gaussians=8, num_colors=8, num_points_per_gaussian=1000, std_dev=0.1, scale=5, labels_as_state=False, labels_as_context=False):
        self.dim = dim
        self.num_points_per_gaussian = num_points_per_gaussian
        self.num_gaussians = num_gaussians
        self.num_colors = num_colors if num_colors > 0 else 1
        self.std_dev = std_dev
        self.scale = scale
        self.continuous, labels = self.sample_N_concentric_gaussians()
        if labels_as_state: self.discrete = labels.long()
        elif labels_as_context: self.context = labels.long()
        else: pass 

    def sample_N_concentric_gaussians(self):
        angle_step = 2 * np.pi / self.num_gaussians
        positions = []
        labels = []

        for i in range(self.num_gaussians):
            angle = i * angle_step
            center_x = np.cos(angle) 
            center_y = np.sin(angle) 
            normal = torch.distributions.multivariate_normal.MultivariateNormal( torch.zeros(self.dim), math.sqrt(self.std_dev) * torch.eye(self.dim))
            points = normal.sample((self.num_points_per_gaussian,))
            points += np.array([center_x, center_y]) * self.scale
            positions.append(points)
            labels += [i % self.num_colors] * self.num_points_per_gaussian

        positions = np.concatenate(positions, axis=0)
        positions = torch.tensor(positions, dtype=torch.float32)
        labels = np.array(labels)
        labels = torch.tensor(labels)
        idx = torch.randperm(len(labels))
        positions = positions[idx]
        labels = labels[idx]
        return positions, labels
    
    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.num_points_per_gaussian * self.num_gaussians if num_points is None else num_points
        c = self.discrete[:num_points] if hasattr(self, 'discrete') else (self.context[:num_points] if hasattr(self, 'context') else None)
        ax.scatter(self.continuous[:num_points, 0], self.continuous[:num_points, 1], c=c, **kwargs)
        plt.xticks([])
        plt.yticks([])
        ax.axis('equal')

    def __len__(self):
        assert self.continuous.shape[0] == self.num_points_per_gaussian * self.num_gaussians
        return self.num_points_per_gaussian * self.num_gaussians

class TwoMoons:
    def __init__(self, dim=2, num_points_per_moon=1000, std_dev=0.2, labels_as_state=False, labels_as_context=False):
        self.dim = dim
        self.num_points_per_moon = num_points_per_moon
        self.std_dev = std_dev
        self.continuous, labels = self.sample_moons()
        if labels_as_state: self.discrete = labels.long()
        elif labels_as_context: self.context = labels.long()
        else: pass 

    def sample_moons(self):
        positions, labels = generate_moons(2 * self.num_points_per_moon , noise=self.std_dev)
        idx = torch.randperm(len(labels))
        positions = positions[idx]
        labels = labels[idx]
        return positions * 3 - 1, labels
    
    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.num_points_per_moon * 2 if num_points is None else num_points
        c = self.discrete[:num_points] if hasattr(self, 'discrete') else (self.context[:num_points] if hasattr(self, 'context') else None)
        ax.scatter(self.continuous[:num_points, 0], self.continuous[:num_points, 1], c=c, **kwargs)
        plt.xticks([])
        plt.yticks([])
        ax.axis('equal')
                
    def __len__(self):
        assert self.continuous.shape[0] == self.num_points_per_moon * 2
        return self.num_points_per_moon * 2

class StdGauss:
    def __init__(self, dim=2, num_colors=1, num_points=1000, std_dev=0.1, labels_as_state=False, labels_as_context=False, pizza_slice=False):
        self.dim = dim
        self.num_points = num_points
        self.num_colors = num_colors if num_colors > 0 else 1   
        self.std_dev = std_dev
        if pizza_slice:
            self.continuous, labels = self.sample_N_pizza()
        else:
            self.continuous = torch.randn(num_points, dim) * std_dev
            labels = np.random.randint(0, num_colors, num_points)
        if labels_as_state: self.discrete = labels
        elif labels_as_context: self.context = labels
        else: pass 

    def sample_N_pizza(self):
        angle_step = 2 * np.pi / self.num_colors
        positions = []
        labels = []

        for i in range(self.num_colors):
            angle_start = i * angle_step
            angle_end = (i + 1) * angle_step
            angles = np.random.uniform(angle_start, angle_end, self.num_points_per_color)
            radii = np.abs(np.random.normal(0, self.std_dev, self.num_points_per_color))  # Ensure radii are positive
            points = np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=1)
            positions.append(points)
            labels += [i] * self.num_points_per_color

        positions = np.concatenate(positions, axis=0)
        positions = torch.tensor(positions, dtype=torch.float32)
        labels = np.array(labels)
        labels = torch.tensor(labels, dtype=torch.long)
        return positions, labels

    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.num_points if num_points is None else num_points
        c = self.discrete[:num_points] if hasattr(self, 'discrete') else (self.context[:num_points] if hasattr(self, 'context') else None)
        ax.scatter(self.continuous[:num_points, 0], self.continuous[:num_points, 1], c=c, **kwargs)
        plt.xticks([])
        plt.yticks([])
        ax.axis('equal')

    def __len__(self):
        assert self.continuous.shape[0] == self.num_points_per_color * self.num_colors
        return self.num_points_per_color * self.num_colors