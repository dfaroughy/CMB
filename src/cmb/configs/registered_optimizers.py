from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from cmb.utils.schedulers import CosineAnnealingWarmup, NoScheduler

optimizers = {'Adam': Adam,
              'AdamW': AdamW,
              'SGD': SGD}

schedulers = {'CosineAnnealingLR': CosineAnnealingLR,
              'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts,
              'CosineAnnealingWarmup': CosineAnnealingWarmup,
              'StepLR': StepLR,
              'ExponentialLR': ExponentialLR,
              'NoScheduler': NoScheduler}

