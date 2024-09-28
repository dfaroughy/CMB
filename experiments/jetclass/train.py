from cmb.configs.experiments import Configs
from cmb.datasets.jetclass import JetsClassData
from cmb.models.trainers import GenerativeDynamicsModule

config = Configs('epic_hybrid.yaml') 
jets = JetsClassData(config.data)
epic_cmb = GenerativeDynamicsModule(config, jets)
epic_cmb.train()