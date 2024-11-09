from cmb.configs.utils import Configs
from cmb.datasets.jetclass import JetDataclass
from cmb.models.trainers import GenerativeDynamicsModule

config = Configs('epic_cmb.yaml') 
jets = JetDataclass(config.data, task='train')
epic_cmb = GenerativeDynamicsModule(config, jets)
epic_cmb.train()