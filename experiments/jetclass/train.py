import torch
import matplotlib.pyplot as plt

from cmb.configs.experiments import Configs

from cmb.datasets.jetclass import CouplingData
from cmb.datasets.jetclass import ParticleClouds

from cmb.dynamics.cfm import ConditionalFlowMatching
from cmb.models.architectures.epic import EPiC
from cmb.models.pipelines import Pipeline
from cmb.models.trainers import CMBTrainer

#...get experiment configuratio:
config = Configs('epic.yaml')

#...define data and model:
jets = CouplingData(config=config.data, standardize=False)
dynamics = ConditionalFlowMatching(config.dynamics)
epic = EPiC(config)

#...tarin model:
model = CMBTrainer(config, dynamics, epic, jets)
model.train()

#...generation pipline:
num_gen_jets = 10000
pipeline = Pipeline(trained_model=model, config=config)
test = CouplingData(config.data)
pipeline.generate_samples(source_continuous=test.source.continuous[:num_gen_jets], mask=test.source.mask[:num_gen_jets])

#...store results:

generated = torch.cat([pipeline.paths[-1], test.source.mask[:num_gen_jets]], dim=-1)
jets_generated = ParticleClouds(generated) 

n=100
fig, ax = plt.subplots(1,3, figsize=(6,2))
jets.source.display_cloud(idx=n, scale_marker=100, ax=ax[0], color='darkblue')
jets_generated.display_cloud(idx=n, scale_marker=100, ax=ax[1], color='purple')
jets.target.display_cloud(idx=n, scale_marker=100, ax=ax[2], color='darkred')
plt.tight_layout()
plt.savefig(model.workdir / 'particle_discplays.png')

#...plots
fig, ax = plt.subplots(3, 3, figsize=(12,8))

binrange_0, binwidth_0 = (-0.1, 1), 0.005
binrange_1, binwidth_1 = (-1, 1), 0.01
binrange_2, binwidth_2 = (-1, 1), 0.01

jets_generated.histplot('pt_rel', binrange=binrange_0, binwidth=binwidth_0, ax=ax[0,0], color='r', log_scale=(False, True), fill=False, stat='density',lw=0.75, label='generated')
jets.target.histplot('pt_rel', binrange=binrange_0, binwidth=binwidth_0, ax=ax[0,0], log_scale=(False, True),  color='k', stat='density', alpha=0.3, lw=0.3, label='target')
jets_generated.histplot('eta_rel', binrange=binrange_1, binwidth=binwidth_1, ax=ax[0,1], log_scale=(False, True), color='r', fill=False, stat='density', lw=0.75, label='generated')
jets.target.histplot('eta_rel', binrange=binrange_1, binwidth=binwidth_1, ax=ax[0,1],   log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')
jets_generated.histplot('phi_rel', binrange=binrange_2, binwidth=binwidth_2, ax=ax[0,2], log_scale=(False, True), color='r', fill=False, stat='density',  lw=0.75, label='generated')
jets.target.histplot('phi_rel',binrange=binrange_2, binwidth=binwidth_2, ax=ax[0,2],  log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')

jets_generated.histplot('pt_rel', idx=0, binrange=binrange_0, binwidth=binwidth_0, ax=ax[1,0], color='r', log_scale=(False, True), fill=False, stat='density',lw=0.75, label='generated')
jets.target.histplot('pt_rel', idx=0, binrange=binrange_0, binwidth=binwidth_0, ax=ax[1,0], log_scale=(False, True),  color='k', stat='density', alpha=0.3, lw=0.3, label='target')
jets_generated.histplot('eta_rel', idx=0, binrange=binrange_1, binwidth=binwidth_1, ax=ax[1,1], log_scale=(False, True), color='r', fill=False, stat='density', lw=0.75, label='generated')
jets.target.histplot('eta_rel', idx=0, binrange=binrange_1, binwidth=binwidth_1, ax=ax[1,1],   log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')
jets_generated.histplot('phi_rel', idx=0, binrange=binrange_2, binwidth=binwidth_2, ax=ax[1,2], log_scale=(False, True), color='r', fill=False, stat='density',  lw=0.75, label='generated')
jets.target.histplot('phi_rel', idx=0, binrange=binrange_2, binwidth=binwidth_2, ax=ax[1,2],  log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')

jets_generated.histplot('pt_rel', idx=20, binrange=binrange_0, binwidth=binwidth_0, ax=ax[2,0], color='r', log_scale=(False, True), fill=False, stat='density',lw=0.75, label='generated')
jets.target.histplot('pt_rel', idx=20, binrange=binrange_0, binwidth=binwidth_0, ax=ax[2,0], log_scale=(False, True),  color='k', stat='density', alpha=0.3, lw=0.3, label='target')
jets_generated.histplot('eta_rel', idx=20, binrange=binrange_1, binwidth=binwidth_1, ax=ax[2,1], log_scale=(False, True), color='r', fill=False, stat='density', lw=0.75, label='generated')
jets.target.histplot('eta_rel', idx=20, binrange=binrange_1, binwidth=binwidth_1, ax=ax[2,1],   log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')
jets_generated.histplot('phi_rel', idx=20, binrange=binrange_2, binwidth=binwidth_2, ax=ax[2,2], log_scale=(False, True), color='r', fill=False, stat='density',  lw=0.75, label='generated')
jets.target.histplot('phi_rel', idx=20, binrange=binrange_2, binwidth=binwidth_2, ax=ax[2,2],  log_scale=(False, True), color='k', stat='density', alpha=0.3, lw=0.3, label='target')

ax[0,0].legend()
plt.tight_layout()
plt.savefig(model.workdir / 'particle_features.png')
