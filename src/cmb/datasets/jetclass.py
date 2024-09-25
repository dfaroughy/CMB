import numpy as np
import torch
import awkward as ak
import fastjet
import vector
import uproot
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from matplotlib.lines import Line2D
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

from cmb.datasets.utils import extract_features

vector.register_awkward()

class JetsBoundaryData:
    ''' class that samples from the source-target coupling q(x_0, x_1)
    '''
    def __init__(self, config: dataclass, standardize: bool=False):
        
        N = config.target.num_jets

        if config.target.name == 'tops':
            self.target = ParticleClouds(config.target.path, 
                                         min_num_particles=config.min_num_particles, 
                                         max_num_particles=config.max_num_particles, 
                                         num_jets=N,
                                         discrete_features=bool(config.dim.features.discrete))
    
        if config.source.name == 'beta-gauss':
            self.source = PointClouds(num_clouds=len(self.target), 
                                      max_num_particles=config.max_num_particles, 
                                      discrete_features=bool(config.dim.features.discrete), 
                                      masks_like=self.target,
                                      noise_type='beta-gauss',
                                      gauss_std=config.source.gauss_std,
                                      concentration=config.source.concentration,
                                      initial_state_configuration=config.source.initial_state_configuration)
            if standardize: 
                self.source.preprocess()


class ParticleClouds:
    def __init__(self, 
                 dataset,
                 num_jets=None,
                 min_num_particles=0,
                 max_num_particles=128,
                 discrete_features=False):
        
        self.min_num_particles = min_num_particles
        self.max_num_particles = max_num_particles

        #...get particle data
        data = extract_features(dataset)
        
        #...pt order data
        idx = torch.argsort(data[...,0], dim=1, descending=True)
        data_pt_sorted = torch.gather(data, 1, idx.unsqueeze(-1).expand(-1, -1, data.size(2)))
        
        #...get continuous features
        self.continuous = data_pt_sorted[...,0:3]
        self.mask = data_pt_sorted[...,-1].long()
        self.pt_rel = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]
        self.multiplicity = torch.sum(self.mask, dim=1)
        self.mask = self.mask.unsqueeze(-1)

        #...get discrete features
        if discrete_features:
            self.flavor = data_pt_sorted[...,3:11]
            self.discrete = torch.argmax(self.flavor, dim=2) # (0: isPhoton, 1: isNeutralHadron, 2: isNegHadron, 3: isPosHadron, 4: isElectron, 5: isAntiElectron, 6: isMuon, 7: isAntiMuon)

    def __len__(self):
        return self.continuous.shape[0]

    def summary_stats(self):
        data = self.continuous[self.mask.squeeze(-1) > 0]
        return {'mean': data.mean(0).numpy(),
                'std': data.std(0).numpy(),
                'min': data.min(0),
                'max': data.max(0)}
    
    def preprocess(self, scale=1.0):
        stats = self.summary_stats()
        self.continuous = (self.continuous - stats['mean']) / (stats['std'] * scale) 
        self.continuous = self.continuous * self.mask
        self.pt_rel = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    def postprocess(self, stats, scale=1.0):
        self.continuous = (self.continuous * stats['std'] * scale) + stats['mean']
        self.pt_rel = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    def histogram(self, features='pt_rel', num_bins=100, density=True, use_quantiles=False):
        mask = self.mask.squeeze(-1) > 0
        x = getattr(self, features)[mask]
        bins = np.quantile(x, np.linspace(0.001, 0.999, num_bins)) if use_quantiles else num_bins
        return np.histogram(x, density=density, bins=bins)[0]

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = self.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        h2 = reference.constituents.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        return scipy.stats.entropy(h1, h2)  

    def Wasserstein1D(self, feature, reference):
        mask = self.mask.squeeze(-1) > 0
        x = getattr(self, feature)[mask]
        y = getattr(reference.constituents, feature)[mask]
        return scipy.stats.wasserstein_distance(x, y)

    def histplot(self, features='pt_rel', xlim=None, ylim=None, xlabel=None, ylabel=None, idx=None, figsize=(3,3), ax=None, **kwargs):
        mask = self.mask.squeeze(-1) > 0
        x = getattr(self, features)[mask] if idx is None else getattr(self, features)[:,idx]
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        sns.histplot(x=x.numpy(), element="step", ax=ax, **kwargs) 
        ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(self, idx, scale_marker=1.0, ax=None, figsize=(3,3) , facecolor='whitesmoke', color='darkblue', title_box_anchor=(1.025,1.125)):
            eta = self.eta_rel[idx].numpy()
            phi = self.phi_rel[idx].numpy()
            pt = 3 * scale_marker * self.pt_rel[idx].numpy()
            mask = list(self.mask.squeeze(-1)[idx].numpy()>0)
            pt = pt[mask] 
            eta = eta[mask] 
            phi = phi[mask] 

            if ax is None: _, ax = plt.subplots(figsize=figsize)            
            
            if hasattr(self, 'discrete'):

                flavor = self.discrete[idx].numpy()
                flavor = flavor[mask]

                ax.scatter(eta[flavor==0], phi[flavor==0], marker='o', s=pt[flavor==0], color='gold', alpha=0.5, label=r'$\gamma$')
                ax.scatter(eta[flavor==1], phi[flavor==1], marker='o', s=pt[flavor==1], color='red', alpha=0.5,  label=r'$h^{0}$')
                ax.scatter(eta[flavor==2], phi[flavor==2], marker='^', s=pt[flavor==2], color='darkred', alpha=0.5,  label=r'$h^{-}$')
                ax.scatter(eta[flavor==3], phi[flavor==3], marker='v', s=pt[flavor==3], color='darkred', alpha=0.5,  label=r'$h^{+}$')
                ax.scatter(eta[flavor==4], phi[flavor==4], marker='^', s=pt[flavor==4], color='blue', alpha=0.5,  label=r'$e^{-}$')
                ax.scatter(eta[flavor==5], phi[flavor==5], marker='v', s=pt[flavor==5], color='blue', alpha=0.5,  label=r'$e^{+}$')
                ax.scatter(eta[flavor==6], phi[flavor==6], marker='^', s=pt[flavor==6], color='green', alpha=0.5,  label=r'$\mu^{-}$')
                ax.scatter(eta[flavor==7], phi[flavor==7], marker='v', s=pt[flavor==7], color='green', alpha=0.5,  label=r'$\mu^{+}$')

                # Define custom legend markers
                h1 = Line2D([0], [0], marker='o', markersize=2, alpha=0.5, color='gold', linestyle='None')
                h2 = Line2D([0], [0], marker='o', markersize=2, alpha=0.5, color='red', linestyle='None')
                h3 = Line2D([0], [0], marker='^', markersize=2, alpha=0.5, color='darkred', linestyle='None')
                h4 = Line2D([0], [0], marker='v', markersize=2, alpha=0.5, color='darkred', linestyle='None')
                h5 = Line2D([0], [0], marker='^', markersize=2, alpha=0.5, color='blue', linestyle='None')
                h6 = Line2D([0], [0], marker='v', markersize=2, alpha=0.5, color='blue', linestyle='None')
                h7 = Line2D([0], [0], marker='^', markersize=2, alpha=0.5, color='green', linestyle='None')
                h8 = Line2D([0], [0], marker='v', markersize=2, alpha=0.5, color='green', linestyle='None')

                plt.legend([h1, h2, h3, h4, h5, h6, h7, h8], 
                        [r'$\gamma$', r'$h^0$', r'$h^-$', r'$h^-$', r'$e^-$', r'$e^+$', r'$\mu^{-}$', r'$\mu^{+}$'], 
                        loc="upper right", 
                        markerscale=2, 
                        scatterpoints=1, 
                        fontsize=7,  
                        frameon=False,
                        ncol=8,
                        bbox_to_anchor=title_box_anchor,
                        handletextpad=-0.5,  
                        columnspacing=.1) 
            else:
                ax.scatter(eta, phi, marker='o', s=pt, color=color, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(facecolor)  # Set the same color for the axis background

class PointClouds:
    def __init__(self, 
                num_clouds, 
                max_num_particles=128,
                masks_like=None,
                discrete_features=False,
                noise_type='gauss',
                concentration=[1.,3.],
                initial_state_configuration='random',
                gauss_std=0.1):
                
        self.num_clouds = num_clouds
        self.max_num_particles = max_num_particles 

        if noise_type=='beta-gauss':
            a, b = torch.tensor(concentration[0]), torch.tensor(concentration[1])
            pt = Beta(a, b).sample((num_clouds, max_num_particles, 1))
            eta_phi = torch.randn((num_clouds, max_num_particles, 2)) * gauss_std
            continuous = torch.cat([pt, eta_phi], dim=2)
        elif noise_type=='gauss':
            continuous = torch.randn((num_clouds, max_num_particles, 3))
        else:
            raise ValueError('Noise type not recognized. Choose between "gauss" and "beta-gauss".')
        
        idx = torch.argsort(continuous[...,0], dim=1, descending=True)
        self.continuous = torch.gather(continuous, 1, idx.unsqueeze(-1).expand(-1, -1, continuous.size(2)))
        self.pt_rel = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]
        self.sample_masks(masks_like=masks_like)

        if discrete_features:
            if initial_state_configuration == 'uniform':
                self.discrete = np.random.choice([0,1,2,3,4,5,6,7], size=(num_clouds, max_num_particles))
            else:
                self.discrete = np.ones((num_clouds, max_num_particles)) * initial_state_configuration
            self.discrete = torch.tensor(self.discrete).long()

    def __len__(self):
        return self.num_clouds 

    def sample_masks(self, masks_like=None):
        ''' Sample masks from a multiplicity distribution of target 'masks_like'.
        '''
        if masks_like is None:
            # If no reference histogram is provided, use full masks
            self.mask = torch.ones_like(self.pt).unsqueeze(-1)
        else:
            max_val = masks_like.max_num_particles
            hist_values, bin_edges = np.histogram(masks_like.multiplicity, bins=np.arange(0, max_val + 2, 1), density=True)
            bin_edges[0] = np.floor(bin_edges[0])   # Ensure lower bin includes the floor of the lowest value
            bin_edges[-1] = np.ceil(bin_edges[-1])  # Extend the upper bin edge to capture all values
            
            histogram = torch.tensor(hist_values, dtype=torch.float)
            probs = histogram / histogram.sum()
            cat = Categorical(probs)
            self.multiplicity = cat.sample((len(self),))  
            
            # Initialize masks and apply the sampled multiplicities
            masks = torch.zeros((len(self), self.max_num_particles))
            for i, n in enumerate(self.multiplicity):
                masks[i, :n] = 1  
            self.mask = masks.long()
            self.mask = self.mask.unsqueeze(-1)

    def histplot(self, features='pt_rel', xlim=None, ylim=None, xlabel=None, ylabel=None, idx=None, figsize=(3,3), ax=None, **kwargs):
        mask = self.mask.squeeze(-1) > 0
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        if isinstance(features, tuple):
            x = getattr(self, features[0])[mask] if idx is None else getattr(self, features[0])[:,idx]
            y = getattr(self, features[1])[mask] if idx is None else getattr(self, features[1])[:,idx]
            sns.histplot(x=x.numpy(), y=y.numpy(), ax=ax, **kwargs)
            ax.set_xlabel(features[0] if xlabel is None else xlabel)
            ax.set_ylabel(features[1] if ylabel is None else ylabel)
        else:
            x = getattr(self, features)[mask] if idx is None else getattr(self, features)[:,idx]
            sns.histplot(x=x.numpy(), element="step", ax=ax, **kwargs) 
            ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def summary_stats(self):
        data = self.continuous[self.mask.squeeze(-1) > 0]
        return {'mean': data.mean(0).numpy(),
                'std': data.std(0).numpy(),
                'min': data.min(0),
                'max': data.max(0)}
    
    def preprocess(self, scale=1.0):
        stats = self.summary_stats()
        self.continuous = (self.continuous - stats['mean']) / (stats['std'] * scale) 
        self.continuous = self.continuous * self.mask
        self.pt_rel = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    def postprocess(self, stats, scale=1.0):
        self.continuous = (self.continuous * stats['std'] * scale) + stats['mean']
        self.pt_rel = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    def display_cloud(self, idx, scale_marker=1.0, ax=None, figsize=(3,3) , facecolor='whitesmoke', color='darkblue', title_box_anchor=(1.025,1.125)):
            eta = self.eta_rel[idx].numpy()
            phi = self.phi_rel[idx].numpy()
            pt = 3 * scale_marker * self.pt_rel[idx].numpy()
            mask = list(self.mask.squeeze(-1)[idx].numpy()>0)
            pt = pt[mask] 
            eta = eta[mask] 
            phi = phi[mask] 

            if ax is None: _, ax = plt.subplots(figsize=figsize)            
            
            if hasattr(self, 'discrete'):

                flavor = self.discrete[idx].numpy()
                flavor = flavor[mask]

                ax.scatter(eta[flavor==0], phi[flavor==0], marker='o', s=pt[flavor==0], color='gold', alpha=0.5, label=r'$\gamma$')
                ax.scatter(eta[flavor==1], phi[flavor==1], marker='o', s=pt[flavor==1], color='red', alpha=0.5,  label=r'$h^{0}$')
                ax.scatter(eta[flavor==2], phi[flavor==2], marker='^', s=pt[flavor==2], color='darkred', alpha=0.5,  label=r'$h^{-}$')
                ax.scatter(eta[flavor==3], phi[flavor==3], marker='v', s=pt[flavor==3], color='darkred', alpha=0.5,  label=r'$h^{+}$')
                ax.scatter(eta[flavor==4], phi[flavor==4], marker='^', s=pt[flavor==4], color='blue', alpha=0.5,  label=r'$e^{-}$')
                ax.scatter(eta[flavor==5], phi[flavor==5], marker='v', s=pt[flavor==5], color='blue', alpha=0.5,  label=r'$e^{+}$')
                ax.scatter(eta[flavor==6], phi[flavor==6], marker='^', s=pt[flavor==6], color='green', alpha=0.5,  label=r'$\mu^{-}$')
                ax.scatter(eta[flavor==7], phi[flavor==7], marker='v', s=pt[flavor==7], color='green', alpha=0.5,  label=r'$\mu^{+}$')

                # Define custom legend markers
                h1 = Line2D([0], [0], marker='o', markersize=2, alpha=0.5, color='gold', linestyle='None')
                h2 = Line2D([0], [0], marker='o', markersize=2, alpha=0.5, color='red', linestyle='None')
                h3 = Line2D([0], [0], marker='^', markersize=2, alpha=0.5, color='darkred', linestyle='None')
                h4 = Line2D([0], [0], marker='v', markersize=2, alpha=0.5, color='darkred', linestyle='None')
                h5 = Line2D([0], [0], marker='^', markersize=2, alpha=0.5, color='blue', linestyle='None')
                h6 = Line2D([0], [0], marker='v', markersize=2, alpha=0.5, color='blue', linestyle='None')
                h7 = Line2D([0], [0], marker='^', markersize=2, alpha=0.5, color='green', linestyle='None')
                h8 = Line2D([0], [0], marker='v', markersize=2, alpha=0.5, color='green', linestyle='None')

                plt.legend([h1, h2, h3, h4, h5, h6, h7, h8], 
                        [r'$\gamma$', r'$h^0$', r'$h^-$', r'$h^+$', r'$e^-$', r'$e^+$', r'$\mu^{-}$', r'$\mu^{+}$'], 
                        loc="upper right", 
                        markerscale=2, 
                        scatterpoints=1, 
                        fontsize=7,  
                        frameon=False,
                        ncol=8,
                        bbox_to_anchor=title_box_anchor,
                        handletextpad=-0.5,  
                        columnspacing=.1) 
            else:
                ax.scatter(eta, phi, marker='o', s=pt, color=color, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(facecolor)  # Set the same color for the axis background

