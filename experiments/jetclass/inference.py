import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from cmb.configs.experiments import Configs
from cmb.datasets.jetclass import JetsBoundaryData
from cmb.models.trainers import CMBTrainer
from cmb.models.architectures.epic import HybridEPiC
from cmb.dynamics.cmb import ConditionalMarkovBridge, BatchOTCMB

config = Configs('epic_hybrid.yaml')

config.train.device = 'cuda:2'
config.model.num_blocks = 8
config.model.dim.embed.time = 32
config.model.dim.embed.features.continuous = 32
config.model.dim.embed.features.discrete = 32

epic = HybridEPiC(config)
dynamics = BatchOTCMB(config)
epic_cmb = CMBTrainer(config, dynamics, epic)
epic_cmb.load(path='/home/df630/CMB/results/JetClass/beta-gauss_to_tops_ConditionalFlowMatching_HybridEPiC_2024.09.26_02h15', model='best')

jets = JetsBoundaryData(config=config.data, num_source_jets=2000, standardize=False)
