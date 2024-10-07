from cmb.configs.utils import Configs
from cmb.datasets.jetclass import JetsClassData, ParticleClouds
from cmb.models.trainers import GenerativeDynamicsModule

from cmb.configs.utils import Configs
from cmb.datasets.jetclass import JetsClassData

config = Configs('epic_hybrid.yaml') 
jets = JetsClassData(config.data, task='train')
epic_cmb = GenerativeDynamicsModule(config, jets)
epic_cmb.train()