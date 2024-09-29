import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from cmb.datasets.jetclass import JetsClassData, ParticleClouds
from cmb.models.trainers import GenerativeDynamicsModule

###########################################################################################################################################
path = '/global/homes/d/dfarough/CMB/results/JetClass/beta-gauss_to_tops_ConditionalMarkovBridge_HybridEPiC_2024.09.29_04h20/config.yaml'
###########################################################################################################################################

epic_cmb = GenerativeDynamicsModule(config=path)
epic_cmb.load()

epic_cmb.config.data.source.test.num_jets = 10000
epic_cmb.config.pipeline.num_timesteps = 500

test = JetsClassData(epic_cmb.config.data, test=True)
epic_cmb.generate(source_continuous=test.source.continuous, 
                  source_discrete=test.source.discrete,
                  mask=test.source.mask)

sample = ParticleClouds(epic_cmb.sample, min_num_particles=0, max_num_particles=128, discrete_features=True) 

args_sam = {'stat':'density', 'log_scale':(False, True), 'fill':False, 'color':'darkred', 'lw':0.75, 'label':'t=1'}
args_tar = {'stat':'density', 'log_scale':(False, True), 'fill':True, 'color':'k','lw':0.3, 'alpha':0.2, 'label':'target'}
args_src = {'stat':'density', 'log_scale':(False, True), 'fill':False, 'color':'darkblue','lw':0.75, 'label':'t=0'}

#...Plot 1
mask_target = (test.target.mask > 0).squeeze() 
mask_source= (test.source.mask > 0).squeeze() 
mask_sample = (sample.mask > 0).squeeze()

fig, ax = plt.subplots(1, 3, figsize=(10,3))

sample.histplot('pt_rel', mask=mask_sample, binrange=(-.1, .75), binwidth=0.005, xlabel=r'particle $p_t^{\rm rel}$', ax=ax[0], **args_sam)
test.target.histplot('pt_rel', mask=mask_target,  binrange=(-.1, .75), binwidth=0.005, xlabel=r'particle $p_t^{\rm rel}$',ax=ax[0], **args_tar)
test.source.histplot('pt_rel', mask=mask_source,  binrange=(-.1, .75), binwidth=0.005, xlabel=r'particle $p_t^{\rm rel}$',ax=ax[0], **args_src)

sample.histplot('eta_rel', mask=mask_sample, binrange=(-1, 1), binwidth=0.01, xlabel=r'particle $\Delta \eta$', ax=ax[1], **args_sam)
test.target.histplot('eta_rel', mask=mask_target,  binrange=(-1, 1), binwidth=0.01, xlabel=r'particle $\Delta \eta$', ax=ax[1],  **args_tar)
test.source.histplot('eta_rel', mask=mask_source, binrange=(-1, 1), binwidth=0.01, xlabel=r'particle $\Delta \eta$', ax=ax[1],  **args_src)

sample.histplot('phi_rel', mask=mask_sample, binrange=(-1, 1), binwidth=0.01, xlabel=r'particle $\Delta \phi$', ax=ax[2], **args_sam)
test.target.histplot('phi_rel', mask=mask_target,  binrange=(-1, 1), binwidth=0.01, xlabel=r'particle $\Delta \phi$', ax=ax[2],  **args_tar)
test.source.histplot('phi_rel', mask=mask_source, binrange=(-1, 1), binwidth=0.01, xlabel=r'particle $\Delta \phi$', ax=ax[2],  **args_src)

ax[0].legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'particle_distributions.png')
plt.show()

#...Plot 2

fig, ax = plt.subplots(1, 1, figsize=(4,3))
                                  
sns.histplot(test.target.discrete[mask_target].squeeze(), binrange=(-0.1, 7.1), element='step', discrete=True, **args_tar)
sns.histplot(sample.discrete[mask_sample].squeeze(), binrange=(-0.1, 7.1), element='step', discrete=True, **args_sam)
sns.histplot(test.source.discrete[mask_source].squeeze(), binrange=(-0.1, 7.1), element='step', discrete=True, **args_src)

ax.legend(loc='upper right', fontsize=8)
ax.set_xlabel('Particle Flavor')
ax.set_xticks(np.arange(8))
ax.set_xticklabels([r'$\gamma$', r'$h^0$', r'$h^-$', r'$h^+$', r'$e^-$', r'$e^+$', r'$\mu^-$', r'$\mu^+$'])
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'flavor_fraction.png')
plt.show()

#...Plot 3

fig, ax = plt.subplots(2, 4, figsize=(11, 5))

dic = {0: r'$\gamma$', 1: r'$h^0$', 2: r'$h^-$', 3: r'$h^+$', 4: r'$e^-$', 5: r'$e^+$', 6: r'$\mu^-$', 7: r'$\mu^+$'}

for n in [0, 1, 2, 3]:
    sample_counts = (sample.discrete == n) * mask_sample.unsqueeze(-1)
    sample_counts = sample_counts.sum(dim=1)

    target_counts = (test.target.discrete == n) * mask_target.unsqueeze(-1)
    target_counts = target_counts.sum(dim=1)

    source_counts = (test.source.discrete == n) * mask_source.unsqueeze(-1)
    source_counts = source_counts.sum(dim=1)

    sns.histplot(target_counts.squeeze(), discrete=True, ax=ax[0, n], element='step', **args_tar)  # black for target
    sns.histplot(source_counts.squeeze(), discrete=True, ax=ax[0, n], element='step', **args_src)  # darkblue for source
    sns.histplot(sample_counts.squeeze(), discrete=True, ax=ax[0, n], element='step', **args_sam)  # darkred for sample

    ax[0, n].set_xlabel(f'{dic[n]} multiplicities')

for n in [0, 1, 2, 3]:
    sample_counts = (sample.discrete == 4 + n).sum(dim=1)
    target_counts = (test.target.discrete == 4 + n).sum(dim=1)
    source_counts = (test.source.discrete == 4 + n).sum(dim=1)

    sns.histplot(target_counts.squeeze(), discrete=True, ax=ax[1, n], element='step', **args_tar)  # black for target
    sns.histplot(source_counts.squeeze(), discrete=True, ax=ax[1, n], element='step', **args_src)  # darkblue for source
    sns.histplot(sample_counts.squeeze(), discrete=True, ax=ax[1, n], element='step', **args_sam)  # darkred for sample

    ax[1, n].set_xlabel(f'{dic[4 + n]} multiplicities')

ax[0, 0].legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'flavor_multiplicities.png')
plt.show()


#...Plot 4

sample_total_charge = (sample.discrete == 2).sum(dim=1) - (sample.discrete == 3).sum(dim=1) + (sample.discrete == 4).sum(dim=1) - (sample.discrete == 5).sum(dim=1) + (sample.discrete == 6).sum(dim=1) - (sample.discrete == 7).sum(dim=1)
target_total_charge = (test.target.discrete == 2).sum(dim=1) - (test.target.discrete == 3).sum(dim=1) + (test.target.discrete == 4).sum(dim=1) - (test.target.discrete == 5).sum(dim=1) + (test.target.discrete == 6).sum(dim=1) - (test.target.discrete == 7).sum(dim=1)
source_total_charge = (test.source.discrete == 2).sum(dim=1) - (test.source.discrete == 3).sum(dim=1) + (test.source.discrete == 4).sum(dim=1) - (test.source.discrete == 5).sum(dim=1) + (test.source.discrete == 6).sum(dim=1) - (test.source.discrete == 7).sum(dim=1)

fig, ax = plt.subplots(1, 1, figsize=(3.5,3))
sns.histplot(sample_total_charge.squeeze(), discrete=True, element='step', **args_sam)
sns.histplot(target_total_charge.squeeze(), discrete=True, element='step', **args_tar)
sns.histplot(source_total_charge.squeeze(), discrete=True, element='step', **args_src)
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'jet_pt.png')
plt.show()

#...Plot 5

sample_jet_pt = sample.continuous[...,0].sum(dim=1)
target_jet_pt = test.target.continuous[...,0].sum(dim=1)
source_jet_pt = test.source.continuous[...,0].sum(dim=1)

fig, ax = plt.subplots(1, 1, figsize=(3.5,3))
sns.histplot(sample_jet_pt, binrange=(0., 3), binwidth=0.05, element='step', **args_sam)
sns.histplot(target_jet_pt, binrange=(0., 3), binwidth=0.05, element='step', **args_tar)
sns.histplot(source_jet_pt, binrange=(0., 3), binwidth=0.05, element='step', **args_src)
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(epic_cmb.workdir / 'jet_pt.png')
plt.show()