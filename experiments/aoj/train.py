from cmb.configs.utils import Configs
from cmb.datasets.jetclass import JetDataclass
from cmb.models.trainers import GenerativeDynamicsModule

config = Configs('configs.yaml') 
jets = JetDataclass(config.data, task='train')
epic_cmb = GenerativeDynamicsModule(config, jets)
epic_cmb.train()