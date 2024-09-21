import torch
import numpy as np
from dataclasses import dataclass
import seaborn as sns
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt
import math

class SampleCoupling:
    ''' Dataclass for sampling coupling between target and source datasets.
    
        targets:
            - Ngaussians
            - moons
            - gaussian 
    '''
    def __init__(self, config):

        self.config = config
        N = config.vocab_size.features if config.dim.features.discrete else config.vocab_size.context

        #...2D Targets:

        if config.target.name == 'Ngaussians': 
            self.target = NGaussians(num_gaussians=N, 
                                     num_colors=N,
                                     num_points_per_gaussian=config.target.num_points // N , 
                                     std_dev=0.1, 
                                     scale=5, 
                                     labels_as_state=bool(config.dim.features.discrete),
                                     labels_as_context=bool(config.dim.context.discrete),
                                     random_colors=config.target.color_pattern == 'random')
            
        elif config.target.name == 'moons': 
            self.target = TwoMoons(num_points_per_moon=config.target.num_points // 2, 
                                   std_dev=0.2, 
                                   labels_as_state=bool(config.dim.features.discrete),
                                   labels_as_context=bool(config.dim.context.discrete),
                                   random_colors=config.target.color_pattern == 'random')
        else:
            raise ValueError('Unknown target dataset.')

        #...2D Sources:
            
        if config.source.name == 'Ngaussians':
            self.source = NGaussians(num_gaussians=N, 
                                     num_colors=N,
                                     num_points_per_gaussian=config.source.num_points // N , 
                                     std_dev=0.1, 
                                     scale=5, 
                                     labels_as_state=bool(config.dim.features.discrete),
                                     labels_as_context=bool(config.dim.context.discrete),
                                     random_colors=config.source.color_pattern == 'random')
            
        elif config.source.name == 'moons':
            self.target = TwoMoons(num_points_per_moon=config.source.num_points//2, 
                                    std_dev=0.2, 
                                    labels_as_state=bool(config.dim.features.discrete),
                                    labels_as_context=bool(config.dim.context.discrete),
                                    random_colors=config.source.color_pattern == 'random')
            
        elif config.source == 'gaussian':
            self.source = StdGauss(num_colors=N,
                                   num_points=config.source.num_points, 
                                   std_dev=0.5, 
                                   labels_as_state=bool(config.dim.features.discrete),
                                   labels_as_context=bool(config.dim.context.discrete),
                                   random_colors=config.source.color_pattern == 'random')
        else:
            raise ValueError('Unknown source dataset.')
        

class NGaussians:
    def __init__(self, dim=2, num_gaussians=8, num_colors=8, num_points_per_gaussian=1000, std_dev=0.1, scale=5, labels_as_state=False, labels_as_context=False, random_colors=False):
        self.dim = dim
        self.num_points_per_gaussian = num_points_per_gaussian
        self.num_gaussians = num_gaussians
        self.N = num_gaussians * num_points_per_gaussian
        self.num_colors = num_colors if num_colors > 0 else 1
        self.std_dev = std_dev
        self.scale = scale
        self.random_colors = random_colors
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
        labels = np.random.randint(0, self.num_gaussians, self.N) if self.random_colors else labels[idx] 
        return positions, torch.tensor(labels, dtype=torch.long)  
    
    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.N if num_points is None else num_points
        c = self.discrete[:num_points] if hasattr(self, 'discrete') else (self.context[:num_points] if hasattr(self, 'context') else None)
        ax.scatter(self.continuous[:num_points, 0], self.continuous[:num_points, 1], c=c, **kwargs)
        plt.xticks([])
        plt.yticks([])
        ax.axis('equal')

    def __len__(self):
        assert self.continuous.shape[0] == self.N
        return self.N

class TwoMoons:
    def __init__(self, dim=2, num_points_per_moon=1000, std_dev=0.2, labels_as_state=False, labels_as_context=False, random_colors=False):
        self.dim = dim
        self.num_points_per_moon = num_points_per_moon
        self.N = num_points_per_moon * 2
        self.std_dev = std_dev
        self.random_colors = random_colors
        self.continuous, labels = self.sample_moons()
        if labels_as_state: self.discrete = labels.long()
        elif labels_as_context: self.context = labels.long()
        else: pass 

    def sample_moons(self):
        positions, labels = generate_moons(self.N, noise=self.std_dev)
        idx = torch.randperm(len(labels))
        positions = positions[idx]
        labels = np.random.randint(0, 2, self.N) if self.random_colors else labels[idx] 
        return positions * 3 - 1, torch.tensor(labels, dtype=torch.long)
    
    def display(self, num_points=None, ax=None, **kwargs):
        num_points = self.N if num_points is None else num_points
        c = self.discrete[:num_points] if hasattr(self, 'discrete') else (self.context[:num_points] if hasattr(self, 'context') else None)
        ax.scatter(self.continuous[:num_points, 0], self.continuous[:num_points, 1], c=c, **kwargs)
        plt.xticks([])
        plt.yticks([])
        ax.axis('equal')
                
    def __len__(self):
        assert self.continuous.shape[0] == self.N
        return self.N

class StdGauss:
    def __init__(self, dim=2, num_colors=1, num_points=1000, std_dev=0.1, labels_as_state=False, labels_as_context=False, random_colors=False):
        self.dim = dim
        self.num_points = num_points
        self.num_colors = num_colors if num_colors > 0 else 1   

        self.std_dev = std_dev
        self.random_colors = random_colors
        if not self.random_colors:
            self.continuous, labels = self.sample_N_pizza()
        else:
            self.continuous = torch.randn(num_points, dim) * std_dev
            labels = np.random.randint(0, num_colors, num_points)
        if labels_as_state: self.discrete = torch.tensor(labels, dtype=torch.long)
        elif labels_as_context: self.context = torch.tensor(labels, dtype=torch.long)
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