import numpy as np
import torch
import awkward as ak
import fastjet
import vector
import uproot
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
vector.register_awkward()

class JetsClassData:
    ''' Class that samples from the source-target coupling for JetClass experiments
    '''
    def __init__(self, config: dataclass):
        
        self.target = ParticleClouds(config.target.path, 
                                    min_num_particles=config.min_num_particles, 
                                    max_num_particles=config.max_num_particles, 
                                    num_jets=config.target.num_jets)

        self.source = PointClouds(num_clouds=len(self.target), 
                                  max_num_particles=config.max_num_particles, 
                                  masks_like=self.target,
                                  noise_type=config.source.name,
                                  gauss_std=config.source.gauss_std,
                                  concentration=config.source.concentration)

class ParticleClouds:
    def __init__(self, 
                 dataset,
                 num_jets=None,
                 min_num_particles=0,
                 max_num_particles=128):
        
        self.min_num_particles = min_num_particles 
        self.max_num_particles = max_num_particles 

        #...get particle data
        data = extract_features(dataset, min_num=min_num_particles, max_num=max_num_particles, num_jets=num_jets)
        
        #...pt order data
        idx = torch.argsort(data[...,0], dim=1, descending=True)
        data_pt_sorted = torch.gather(data, 1, idx.unsqueeze(-1).expand(-1, -1, data.size(2)))
        
        #...get particle features
        self.particles = data_pt_sorted[...,0:3]
        self.mask = data_pt_sorted[...,-1].long()
        self.multiplicity = torch.sum(self.mask, dim=1)
        self.mask = self.mask.unsqueeze(-1)
        
        self.pt_rel = self.particles[...,0] 
        self.eta_rel = self.particles[...,1]
        self.phi_rel = self.particles[...,2]

    def __len__(self):
        return self.particles.shape[0]

    #...plots:

    def histplot(self, features='pt_rel', xlim=None, ylim=None, xlabel=None, ylabel=None, idx=None, figsize=(3,3), ax=None, **kwargs):
        mask = self.mask.squeeze(-1) > 0
        x = getattr(self, features)[mask] if idx is None else getattr(self, features)[:,idx]
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        sns.histplot(x=x.numpy(), element="step", ax=ax, **kwargs) 
        ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(self, idx, scale_marker=1.0, ax=None, figsize=(3,3) , facecolor='whitesmoke', color='darkblue'):
            eta = self.eta_rel[idx].numpy()
            phi = self.phi_rel[idx].numpy()
            pt = scale_marker * self.pt_rel[idx].numpy()
            mask = list(self.mask.squeeze(-1)[idx].numpy()>0)
            pt = pt[mask] 
            eta = eta[mask] 
            phi = phi[mask] 

            if ax is None: _, ax = plt.subplots(figsize=figsize)            
            ax.scatter(eta, phi, marker='o', s=scale_marker*pt, color=color, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(facecolor)  # Set the same color for the axis background

    #...metrics:
            
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


class PointClouds:
    def __init__(self, 
                num_clouds, 
                max_num_particles=128,
                masks_like=None,
                noise_type='gauss',
                concentration=[1., 3.], # just for beta-gauss
                gauss_std=0.1):
                
        self.num_clouds = num_clouds
        self.max_num_particles = max_num_particles 

        if noise_type=='beta-gauss':
            a, b = torch.tensor(concentration[0]), torch.tensor(concentration[1])
            pt = Beta(a, b).sample((num_clouds, max_num_particles, 1))
            eta_phi = torch.randn((num_clouds, max_num_particles, 2)) * gauss_std
            data = torch.cat([pt, eta_phi], dim=2)

        elif noise_type=='gauss':
            data = torch.randn((num_clouds, max_num_particles, 3)) * gauss_std

        else:
            raise ValueError('Noise type not recognized. Choose between "gauss" and "beta-gauss".')
        
        idx = torch.argsort(data[...,0], dim=1, descending=True)
        self.particles = torch.gather(data, 1, idx.unsqueeze(-1).expand(-1, -1, data.size(2)))
        self.pt_rel = self.particles[...,0] 
        self.eta_rel = self.particles[...,1]
        self.phi_rel = self.particles[...,2]
        self.sample_masks(masks_like=masks_like)

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
            
            masks = torch.zeros((len(self), self.max_num_particles))
            for i, n in enumerate(self.multiplicity):
                masks[i, :n] = 1  
            self.mask = masks.long()
            self.mask = self.mask.unsqueeze(-1)

    #...plots:

    def histplot(self, features='pt_rel', xlim=None, ylim=None, xlabel=None, ylabel=None, idx=None, figsize=(3,3), ax=None, **kwargs):
        mask = self.mask.squeeze(-1) > 0
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        x = getattr(self, features)[mask] if idx is None else getattr(self, features)[:,idx]
        sns.histplot(x=x.numpy(), element="step", ax=ax, **kwargs) 
        ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(self, idx, scale_marker=1.0, ax=None, figsize=(3,3) , facecolor='whitesmoke', color='darkblue'):
            eta = self.eta_rel[idx].numpy()
            phi = self.phi_rel[idx].numpy()
            pt = scale_marker * self.pt_rel[idx].numpy()
            mask = list(self.mask.squeeze(-1)[idx].numpy()>0)
            pt = pt[mask] 
            eta = eta[mask] 
            phi = phi[mask] 

            if ax is None: _, ax = plt.subplots(figsize=figsize)            
            ax.scatter(eta, phi, marker='o', s=scale_marker*pt, color=color, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(facecolor)  # Set the same color for the axis background


#...Helper functions:
            
def extract_features(dataset, min_num=0, max_num=128, num_jets=None):
    if isinstance(dataset, list):
        all_data = []
        for data in dataset:
            assert  '.root' in data, 'Input should be a path to a .root file or a tensor'
            data = read_root_file(data)
            features = ['part_ptrel', 'part_etarel', 'part_phirel', 'mask']       
            data = torch.tensor(np.stack([ak.to_numpy(pad(data[feat], min_num=min_num, max_num=max_num)) for feat in features] , axis=1))
            data = torch.permute(data, (0,2,1))   
            all_data.append(data)   
        data = torch.cat(all_data, dim=0)   
        data = data[:num_jets] if num_jets is not None else data
    else:
        assert isinstance(dataset, torch.Tensor), 'Input should be a path to a .root file or a tensor'
        data = dataset[:num_jets] if num_jets is not None else dataset
    return data

def read_root_file(filepath):
    
    """Loads a single .root file from the JetClass dataset.
    """
    x = uproot.open(filepath)['tree'].arrays()
    x['part_pt'] = np.hypot(x['part_px'], x['part_py'])
    x['part_pt_log'] = np.log(x['part_pt'])
    x['part_ptrel'] = x['part_pt'] / x['jet_pt']
    x['part_deltaR'] = np.hypot(x['part_deta'], x['part_dphi'])

    p4 = vector.zip({'px': x['part_px'],
                        'py': x['part_py'],
                        'pz': x['part_pz'],
                        'energy': x['part_energy']})

    x['part_eta'] = p4.eta
    x['part_phi'] = p4.phi
    x['part_etarel'] = p4.eta - x['jet_eta'] 
    x['part_phirel'] = (p4.phi - x['jet_phi'] + np.pi) % (2 * np.pi) - np.pi
    x['mask'] = np.ones_like(x['part_energy']) 
    return x

def pad(a, min_num, max_num, value=0, dtype='float32'):
    assert max_num >= min_num, 'max_num must be >= min_num'
    assert isinstance(a, ak.Array), 'Input must be an awkward array'
    a = a[ak.num(a) >= min_num]
    a = ak.fill_none(ak.pad_none(a, max_num, clip=True), value)
    return ak.values_astype(a, dtype)
