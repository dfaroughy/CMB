import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from cmb.configs.experiments import Configs
from cmb.datasets.jetclass import JetsBoundaryData
from cmb.models.trainers import CMBTrainer
from cmb.models.architectures.epic import HybridEPiC
from cmb.datasets.jetclass import ParticleClouds
from cmb.models.pipelines import Pipeline

from cmb.dynamics.cmb import ConditionalMarkovBridge, BatchOTCMB

config = Configs('epic_hybrid.yaml')
jets = JetsBoundaryData(config=config.data, standardize=False)
epic = HybridEPiC(config)
dynamics = ConditionalMarkovBridge(config)
epic_cmb = CMBTrainer(config, dynamics, epic, jets)
epic_cmb.train()


num_gen_jets = 3000

pipeline = Pipeline(config=config, trained_model=epic_cmb, dynamics=dynamics)
test = JetsBoundaryData(config.data)

pipeline.generate_samples(source_continuous=test.source.continuous[:num_gen_jets], 
                          source_discrete=torch.tensor(test.source.discrete[:num_gen_jets]),
                          mask=test.source.mask[:num_gen_jets])


jets_generated = torch.cat([pipeline.paths[-1], pipeline.jumps[-1].unsqueeze(-1), test.source.mask[:num_gen_jets]], dim=-1)
jets_generated = ParticleClouds(jets_generated, min_num_particles=0, max_num_particles=128, discrete_features=True) 

fig, ax = plt.subplots(1, 3, figsize=(12,3))

jets_generated.histplot('pt_rel', binrange=(-.1, 1), binwidth=0.005, ax=ax[0], color='r', log_scale=(False, True), fill=False, stat='density',lw=0.75, label='generated')
jets.target.histplot('pt_rel', binrange=(-.1, 1), binwidth=0.005, ax=ax[0], log_scale=(False, True),  color='k', stat='density', alpha=0.3, lw=0.3, label='target')

jets_generated.histplot('eta_rel', binrange=(-1, 1), binwidth=0.01, ax=ax[1], log_scale=(False, True), color='r', fill=False, stat='density', lw=0.75, label='generated')
jets.target.histplot('eta_rel', binrange=(-1, 1), binwidth=0.01, ax=ax[1],   log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')

jets_generated.histplot('phi_rel', binrange=(-1, 1), binwidth=0.01, ax=ax[2], log_scale=(False, True), color='r', fill=False, stat='density',  lw=0.75, label='generated')
jets.target.histplot('phi_rel',binrange=(-1, 1), binwidth=0.01, ax=ax[2],  log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')

ax[2].legend()
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'particle_distributions.png')

g, ax = plt.subplots(1, 3, figsize=(12,3))

jets_generated.histplot('pt_rel', idx=0, binrange=(0, 1), binwidth=0.01, ax=ax[0], color='r', log_scale=(False, True), fill=False, stat='density',lw=0.75, label='generated')
jets.target.histplot('pt_rel', idx=0, binrange=(0, 1), binwidth=0.01, ax=ax[0], log_scale=(False, True),  color='k', stat='density', alpha=0.3, lw=0.3, label='target')

jets_generated.histplot('eta_rel', idx=0, binrange=(-1, 1), binwidth=0.02, ax=ax[1], log_scale=(False, True), color='r', fill=False, stat='density', lw=0.75, label='generated')
jets.target.histplot('eta_rel', idx=0,  binrange=(-1, 1), binwidth=0.02, ax=ax[1],  log_scale=(False, True),  color='k', stat='density', alpha=0.3, lw=0.3, label='target')

jets_generated.histplot('phi_rel', idx=0, binrange=(-1, 1), binwidth=0.02, ax=ax[2],log_scale=(False, True),  color='r', fill=False, stat='density',  lw=0.75, label='generated')
jets.target.histplot('phi_rel', idx=0, binrange=(-1, 1), binwidth=0.02, ax=ax[2],  log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')

ax[2].legend()
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'leading_particle_distributions.png')

fig, ax = plt.subplots(1, 1, figsize=(3.5,3))

mask_gen = jets_generated.mask[:num_gen_jets] > 0
mask_data = jets.target.mask > 0

sns.histplot(jets.target.discrete[mask_data.squeeze(-1)], discrete=True, element='step', alpha=0.2, lw=0., color='k', label='target', stat='density', log_scale=(False, True))
sns.histplot(jets_generated.discrete[mask_gen.squeeze(-1)], discrete=True, element='step', fill=False, color='darkred', lw=1,label='generated', stat='density', log_scale=(False, True))
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'flavor_fraction.png')

fig, ax = plt.subplots(2, 4, figsize=(11,5))

dic = {0:r'$\gamma$', 1:r'$h^0$', 2:r'$h^-$', 3:r'$h^+$', 4:r'$e^-$', 5:r'$e^+$', 6:r'$\mu^-$', 7:r'$\mu^+$'}

for n in [0,1,2,3]:
    gen_counts = (jets_generated.discrete == n) * mask_gen.squeeze(-1) 
    gen_counts = gen_counts.sum(dim=1)
    data_counts = (jets.target.discrete== n) * mask_data.squeeze(-1)
    data_counts = data_counts.sum(dim=1)
    
    sns.histplot(data_counts , discrete=True, element='step', alpha=0.2, lw=0., color='k', label='target', stat='density', log_scale=(False, True), ax=ax[0,n])
    sns.histplot(gen_counts, discrete=True, element='step', fill=False, color='darkred', lw=1,label='generated', stat='density', log_scale=(False, True), ax=ax[0,n])
    ax[0,n].set_xlabel(f'{dic[n]} multiplicities')
for n in [0,1,2,3]:
    gen_counts = (jets_generated.discrete == 4+n).sum(dim=1)
    data_counts = (jets.target.discrete== 4+n).sum(dim=1)
    sns.histplot(data_counts , discrete=True, element='step', alpha=0.2, lw=0., color='k', label='target', stat='density', log_scale=(False, True), ax=ax[1,n])
    sns.histplot(gen_counts, discrete=True, element='step', fill=False, color='darkred', lw=1,label='generated', stat='density', log_scale=(False, True), ax=ax[1,n])
    ax[1,n].set_xlabel(f'{dic[4+n]} multiplicities')

plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'flavor_fraction.png')

gen_total_charge = (jets_generated.discrete == 2).sum(dim=1) - (jets_generated.discrete == 3).sum(dim=1) + (jets_generated.discrete == 4).sum(dim=1) - (jets_generated.discrete == 5).sum(dim=1) + (jets_generated.discrete == 6).sum(dim=1) - (jets_generated.discrete == 7).sum(dim=1)
data_total_charge = (jets.target.discrete == 2).sum(dim=1) - (jets.target.discrete == 3).sum(dim=1) + (jets.target.discrete == 4).sum(dim=1) - (jets.target.discrete == 5).sum(dim=1) + (jets.target.discrete == 6).sum(dim=1) - (jets.target.discrete == 7).sum(dim=1)

sns.histplot(gen_total_charge, discrete=True, element='step', fill=False, color='darkred', lw=1,label='generated', stat='density', log_scale=(False, True))
sns.histplot(data_total_charge, discrete=True, element='step', alpha=0.2, lw=0., color='k', label='target', stat='density', log_scale=(False, True))

plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'jet_charge.png')

gen_jet_pt = jets_generated.continuous[...,0].sum(dim=1)
data_jet_pt = jets.target.continuous[...,0].sum(dim=1)

sns.histplot(gen_jet_pt, element='step', fill=False, color='darkred', lw=1,label='generated', stat='density', log_scale=(False, True))
sns.histplot(data_jet_pt,  element='step', alpha=0.2, lw=0., color='k', label='target', stat='density', log_scale=(False, True))
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'jet_pt.png')
