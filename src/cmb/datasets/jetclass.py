import numpy as np
import torch
import awkward as ak
import fastjet
import vector
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

from cmb.datasets.utils import extract_features

vector.register_awkward()

class JetsClassData:
    ''' class that samples from the source-target coupling q(x_0, x_1)
    '''
    def __init__(self, 
                 config: dataclass, 
                 dataset_source=None, 
                 dataset_target=None, 
                 num_jets_source=None, 
                 num_jets_target=None,
                 task='train'):

        assert task=='train' or task=='test', 'Please specify a task: `train`, `test`'

        if task == 'test': 
            print('INFO: Loading test datasets.')
            N_target = config.target.test.num_jets if num_jets_target is None else num_jets_target
            N_source = config.source.test.num_jets if num_jets_source is None else num_jets_source
            dataset_target = config.target.test.path if dataset_target is None else dataset_target
            dataset_source = config.target.test.path if dataset_source is None else dataset_source
            self.test_mode = True

        if task == 'train': 
            print('INFO: Loading train datasets.')
            N_target = config.target.train.num_jets if num_jets_target is None else num_jets_target
            N_source = config.source.train.num_jets if num_jets_source is None else num_jets_source
            dataset_target = config.target.train.path if dataset_target is None else dataset_target
            dataset_source = config.target.test.path if dataset_source is None else dataset_source
            self.test_mode = False

        if (config.target.name in ['qcd', 'tbqq', 'tblv', 'wqq', 'hbb']):
            self.target = ParticleClouds(dataset=dataset_target, 
                                         min_num_particles=config.min_num_particles, 
                                         max_num_particles=config.max_num_particles, 
                                         num_jets=N_target,
                                         discrete_features=bool(config.dim.features.discrete))

        if config.source.name == 'beta-gauss':
            self.source = PointClouds(num_clouds=N_source if self.test_mode  else len(self.target), 
                                      max_num_particles=config.max_num_particles, 
                                      discrete_features=bool(config.dim.features.discrete), 
                                      vocab_size=config.vocab.size.features,
                                      masks_like=self.target,
                                      noise_type='beta-gauss',
                                      gauss_std=config.source.gauss_std,
                                      concentration=config.source.concentration,
                                      init_state_config=config.source.init_state_config)
            
        elif config.source.name == 'gauss':
            self.source = PointClouds(num_clouds=N_source if self.test_mode else len(self.target), 
                                      max_num_particles=config.max_num_particles, 
                                      discrete_features=bool(config.dim.features.discrete), 
                                      vocab_size=config.vocab.size.features,
                                      masks_like=self.target,
                                      noise_type='gauss',
                                      gauss_std=config.source.gauss_std,
                                      concentration=config.source.concentration,
                                      init_state_config=config.source.init_state_config)
            
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
        data = extract_features(dataset, num_jets=num_jets)
        
        #...pt order data
        idx = torch.argsort(data[...,0], dim=1, descending=True)
        data_pt_sorted = torch.gather(data, 1, idx.unsqueeze(-1).expand(-1, -1, data.size(2)))
        
        #...get continuous features
        self.continuous = data_pt_sorted[...,0:3]
        self.mask = data_pt_sorted[...,-1].long()
        self.pt = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]
        self.multiplicity = torch.sum(self.mask, dim=1)
        self.mask = self.mask.unsqueeze(-1)

        #...get discrete features
        if discrete_features:
            self.discrete = data_pt_sorted[..., 3].unsqueeze(-1).long() # (1: isPhoton, 2: isNeutralHadron, 3: isNegHadron, 4: isPosHadron, 5: isElectron, 6: isAntiElectron, 7: isMuon, 8: isAntiMuon)
        
    def __len__(self):
        return self.continuous.shape[0]

    def compute_4mom(self):
        self.px = self.pt * torch.cos(self.phi_rel)
        self.py = self.pt * torch.sin(self.phi_rel)
        self.pz = self.pt * torch.sinh(self.eta_rel)
        self.e = self.pt * torch.cosh(self.eta_rel)

    def flavor_charge_rep(self):
        ''' Get the (flavor, charge) representation of each particle:
            0 -> (0, 0), 1 -> (1, 0), 
                         2 -> (1,-1), 4 -> (2,-1), 6 -> (3, 1)
                         3 -> (1, 1), 5 -> (2, 1), 7 -> (3,-1)
        '''
        flavor = self.discrete.clone()
        charge = self.discrete.clone()
        for n in [1,2,3]: flavor[flavor==n] = 1
        for n in [4,5]:   flavor[flavor==n] = 2
        for n in [6,7]:   flavor[flavor==n] = 3
        for n in [0,1]:   charge[charge==n] = 0
        for n in [2,4,6]: charge[charge==n] = -1
        for n in [3,5,7]: charge[charge==n] = 1
        self.flavor = flavor.squeeze(-1)
        self.charge = charge.squeeze(-1)

    #...data processing methods

    def summary_stats(self):
        mask = self.mask.squeeze(-1) > 0
        data = self.continuous[mask]
        return {'mean': data.mean(0).tolist(),
                'std': data.std(0).tolist(),
                'min': data.min(0).values.tolist(),
                'max': data.max(0).values.tolist()}
    
    def preprocess(self, stats, scale=1.0):
        self.continuous = (self.continuous - torch.tensor(stats['mean'])) / (torch.tensor(stats['std']) * scale) 
        self.continuous *= self.mask
        self.pt = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    def postprocess(self, stats, scale=1.0):
        self.continuous = (self.continuous * torch.tensor(stats['std']) * scale) + torch.tensor(stats['mean'])
        self.continuous *= self.mask
        self.pt = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    #...data visualization methods

    def histplot(self, features='pt', mask=None, xlim=None, ylim=None, xlabel=None, ylabel=None, figsize=(3,3), ax=None, **kwargs):
        x = getattr(self, features)[mask] if mask is not None else getattr(self, features)
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        sns.histplot(x=x.numpy(), element="step", ax=ax, **kwargs) 
        ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(self, idx, scale_marker=1.0, ax=None, figsize=(3,3) , facecolor='whitesmoke', color='darkblue', title_box_anchor=(1.025,1.125)):
            eta = self.eta_rel[idx].numpy()
            phi = self.phi_rel[idx].numpy()
            pt = self.pt[idx].numpy() * scale_marker 
            mask = list(self.mask.squeeze(-1)[idx].numpy()>0)
            pt = pt[mask] 
            eta = eta[mask] 
            phi = phi[mask] 

            if ax is None: _, ax = plt.subplots(figsize=figsize)            
            
            if hasattr(self, 'discrete'):

                flavor = self.discrete.squeeze(-1)[idx].numpy()
                flavor = flavor[mask]

                ax.scatter(eta[flavor==0], phi[flavor==0], marker='o', s=pt[flavor==0], color='gold', alpha=0.5, label=r'$\gamma$')
                ax.scatter(eta[flavor==1], phi[flavor==1], marker='o', s=pt[flavor==1], color='darkred', alpha=0.5,  label=r'$h^{0}$')
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
                vocab_size=None,
                noise_type='gauss',
                concentration=[1.,3.],
                init_state_config='uniform',
                gauss_std=0.1):
                
        self.num_clouds = num_clouds
        self.max_num_particles = max_num_particles 

        if noise_type=='beta-gauss':
            a, b = torch.tensor(concentration[0]), torch.tensor(concentration[1])
            pt = Beta(a, b).sample((num_clouds, max_num_particles, 1))
            eta_phi = torch.randn((num_clouds, max_num_particles, 2)) * gauss_std
            continuous = torch.cat([pt, eta_phi], dim=2)
        elif noise_type=='gauss':
            continuous = torch.randn((num_clouds, max_num_particles, 3)) * gauss_std
        else:
            raise ValueError('Noise type not recognized. Choose between "gauss" and "beta-gauss".')
        self.sample_masks(masks_like=masks_like)
        idx = torch.argsort(continuous[...,0], dim=1, descending=True)
        self.continuous = torch.gather(continuous, 1, idx.unsqueeze(-1).expand(-1, -1, continuous.size(2)))
        self.pt = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

        if discrete_features:
            if init_state_config == 'uniform':
                discrete = np.random.choice(range(vocab_size), size=(num_clouds, max_num_particles))
            elif isinstance(init_state_config, list):
                discrete = np.random.choice(init_state_config, size=(num_clouds, max_num_particles))
            elif isinstance(init_state_config, int):
                discrete = np.ones((num_clouds, max_num_particles)) * init_state_config
            discrete = torch.tensor(discrete).unsqueeze(-1) 
            self.discrete = discrete.long()

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

    def compute_4mom(self):
        self.px = self.pt * torch.cos(self.phi_rel)
        self.py = self.pt * torch.sin(self.phi_rel)
        self.pz = self.pt * torch.sinh(self.eta_rel)
        self.e = self.pt * torch.cosh(self.eta_rel)

    def flavor_charge_rep(self):
        ''' Get the (flavor, charge) representation of each particle:
            0 -> (0, 0), 1 -> (1, 0), 
                         2 -> (1,-1), 4 -> (2,-1), 6 -> (3, 1)
                         3 -> (1, 1), 5 -> (2, 1), 7 -> (3,-1)
        '''
        flavor = self.discrete.clone()
        charge = self.discrete.clone()
        for n in [1,2,3]: flavor[flavor==n] = 1
        for n in [4,5]:   flavor[flavor==n] = 2
        for n in [6,7]:   flavor[flavor==n] = 3
        for n in [0,1]:   charge[charge==n] = 0
        for n in [2,4,6]: charge[charge==n] = -1
        for n in [3,5,7]: charge[charge==n] = 1
        self.flavor = flavor.squeeze(-1)
        self.charge = charge.squeeze(-1)

    #...data processing methods
        
    def summary_stats(self):
        mask = self.mask.squeeze(-1) > 0
        data = self.continuous[mask]
        return {'mean': data.mean(0).tolist(),
                'std': data.std(0).tolist(),
                'min': data.min(0).values.tolist(),
                'max': data.max(0).values.tolist()}
    
    def preprocess(self, stats, scale=1.0):
        self.continuous = (self.continuous - torch.tensor(stats['mean'])) / (torch.tensor(stats['std']) * scale) 
        self.continuous *= self.mask
        self.pt = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    def postprocess(self, stats, scale=1.0):
        self.continuous = (self.continuous * torch.tensor(stats['std']) * scale) + torch.tensor(stats['mean'])
        self.continuous *= self.mask
        self.pt = self.continuous[...,0] 
        self.eta_rel = self.continuous[...,1]
        self.phi_rel = self.continuous[...,2]

    #...data visualization methods

    def histplot(self, features='pt', mask=None, xlim=None, ylim=None, xlabel=None, ylabel=None, idx=None, figsize=(3,3), ax=None, **kwargs):
        x = getattr(self, features)[mask] if mask is not None else getattr(self, features)
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        sns.histplot(x=x.numpy(), element="step", ax=ax, **kwargs) 
        ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(self, idx, scale_marker=1.0, ax=None, figsize=(3,3), facecolor='whitesmoke', color='darkblue', title_box_anchor=(1.025,1.125)):
        eta = self.eta_rel[idx].numpy()
        phi = self.phi_rel[idx].numpy()
        pt = self.pt[idx].numpy() * scale_marker 
        mask = list(self.mask.squeeze(-1)[idx].numpy()>0)
        pt = pt[mask] 
        eta = eta[mask] 
        phi = phi[mask] 

        if ax is None: _, ax = plt.subplots(figsize=figsize)            
        
        if hasattr(self, 'discrete'):

            flavor = self.discrete.squeeze(-1)[idx].numpy()
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


class JetClassHighLevelFeatures:
    def __init__(self, constituents: ParticleClouds):

        self.constituents = constituents

        #...compute jet kinematics:
        self.constituents.compute_4mom()
        self.px = self.constituents.px.sum(axis=-1)
        self.py = self.constituents.py.sum(axis=-1)
        self.pz = self.constituents.pz.sum(axis=-1)
        self.e = self.constituents.e.sum(axis=-1)
        self.pt = torch.clamp_min(self.px**2 + self.py**2, 0).sqrt()
        self.m = torch.clamp_min(self.e**2 - self.px**2 - self.py**2 - self.pz**2, 0).sqrt()
        self.eta = 0.5 * torch.log((self.pt + self.pz) / (self.pt - self.pz))
        self.phi = torch.atan2(self.py, self.px)

        # discrete jet features
        self.constituents.flavor_charge_rep()
        self.Q_total = self.jet_charge(kappa=0.0)
        self.Q_jet = self.jet_charge(kappa=1.0)
        self.multiplicity = torch.sum(self.constituents.mask, dim=1)

        #...subsstructure
        self.R = 0.8
        self.beta = 1.0
        self.use_wta_scheme = False
        self.substructure()

    def histplot(self, features='pt', xlim=None, ylim=None, xlabel=None, ylabel=None, figsize=(3,3), ax=None, **kwargs):
        x = getattr(self, features)
        if isinstance(x, torch.Tensor): x.cpu().numpy()
        if ax is None: 
            _, ax = plt.subplots(figsize=figsize)   
        sns.histplot(x=x, element="step", ax=ax, **kwargs) 
        ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def jet_charge(self, kappa):
        ''' jet charge defined as Q_j^kappa = Sum_i Q_i * (pT_i / pT_jet)^kappa
        '''
        Qjet = self.constituents.charge * (self.constituents.pt)**kappa
        return Qjet.sum(axis=1) / (self.pt**kappa) 
    
    def histplot_multiplicities(self, state=None, xlim=None, ylim=None, xlabel=None, ylabel=None, figsize=(3,3), ax=None, **kwargs):
        if state is not None:
            if isinstance(state, int):
                state = [state]
            multiplicity = torch.zeros(self.constituents.discrete.shape[0], 1)
            for s in state:
                x = (self.constituents.discrete == s) * self.constituents.mask
                multiplicity += x.sum(dim=1)
        else:
            multiplicity = self.multiplicity

        if ax is None: 
            _, ax = plt.subplots(figsize=figsize)   
        sns.histplot(x=multiplicity.squeeze(-1), element="step", ax=ax, discrete=True, **kwargs) 
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def flavor_fractions(self, figsize=(3,3), ax=None, **kwargs):
        if ax is None: 
            _, ax = plt.subplots(figsize=figsize)                                          
        sns.histplot(self.constituents.discrete[self.constituents.mask].squeeze(), binrange=(-0.1, 7.1), element="step", ax=ax, discrete=True, **kwargs)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_xlabel('Particle flavor')
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels([r'$\gamma$', r'$h^0$', r'$h^-$', r'$h^+$', r'$e^-$', r'$e^+$', r'$\mu^-$', r'$\mu^+$'])

    def substructure(self):
        constituents_ak = ak.zip({ "pt": np.array(self.constituents.pt),
                                   "eta": np.array(self.constituents.eta_rel),
                                   "phi": np.array(self.constituents.phi_rel),
                                   "mass": np.zeros_like(np.array(self.constituents.pt))
                                   },
                                  with_name="Momentum4D")

        constituents_ak = ak.mask(constituents_ak, constituents_ak.pt > 0)
        constituents_ak = ak.drop_none(constituents_ak)
        self.constituents_ak = constituents_ak[ak.num(constituents_ak) >= 3]
        if self.use_wta_scheme:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme)
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R)
        print("Clustering jets with fastjet")
        print("Jet definition:", jetdef)
        self.cluster = fastjet.ClusterSequence(constituents_ak, jetdef)
        self.inclusive_jets = self.cluster.inclusive_jets()
        self.exclusive_jets_1 = self.cluster.exclusive_jets(n_jets=1)
        self.exclusive_jets_2 = self.cluster.exclusive_jets(n_jets=2)
        self.exclusive_jets_3 = self.cluster.exclusive_jets(n_jets=3)
        print("Calculating N-subjettiness")
        self._calc_d0()
        self._calc_tau1()
        self._calc_tau2()
        self._calc_tau3()
        self.tau21 = np.ma.divide(self.tau2, self.tau1)
        self.tau32 = np.ma.divide(self.tau3, self.tau2)
        print("Calculating D2")
        # D2 as defined in https://arxiv.org/pdf/1409.6298.pdf
        self.d2 = self.cluster.exclusive_jets_energy_correlator(njets=1, func="d2")

    def _calc_deltaR(self, particles, jet):
        jet = ak.unflatten(ak.flatten(jet), counts=1)
        return particles.deltaR(jet)

    def _calc_d0(self):
        """Calculate the d0 values."""
        self.d0 = ak.sum(self.constituents_ak.pt * self.R**self.beta, axis=1)

    def _calc_tau1(self):
        """Calculate the tau1 values."""
        self.delta_r_1i = self._calc_deltaR(self.constituents_ak, self.exclusive_jets_1[:, :1])
        self.pt_i = self.constituents_ak.pt
        self.tau1 = ak.sum(self.pt_i * self.delta_r_1i**self.beta, axis=1) / self.d0

    def _calc_tau2(self):
        """Calculate the tau2 values."""
        delta_r_1i = self._calc_deltaR(self.constituents_ak, self.exclusive_jets_2[:, :1])
        delta_r_2i = self._calc_deltaR(self.constituents_ak, self.exclusive_jets_2[:, 1:2])
        self.pt_i = self.constituents_ak.pt
        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau2 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self):
        """Calculate the tau3 values."""
        delta_r_1i = self._calc_deltaR(self.constituents_ak, self.exclusive_jets_3[:, :1])
        delta_r_2i = self._calc_deltaR(self.constituents_ak, self.exclusive_jets_3[:, 1:2])
        delta_r_3i = self._calc_deltaR(self.constituents_ak, self.exclusive_jets_3[:, 2:3])
        self.pt_i = self.constituents_ak.pt
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                    delta_r_3i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau3 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def histogram(self, features='pt', density=True, num_bins=100, use_quantiles=False):
        x = getattr(self, features)
        bins = np.quantile(x, np.linspace(0.001, 0.999, num_bins)) if use_quantiles else num_bins
        return np.histogram(x, density=density, bins=bins)[0]

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = self.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        h2 = reference.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        return scipy.stats.entropy(h1, h2)  

    def Wassertein1D(self, feature, reference):
        x = getattr(self, feature)
        y = getattr(reference, feature)
        return scipy.stats.wasserstein_distance(x, y)