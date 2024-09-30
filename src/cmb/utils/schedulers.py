
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup, T_max):
        self.warmup = warmup
        self.T_max = T_max
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.T_max))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class NoScheduler:
    def __init__(self, optimizer): pass
    def step(self): pass    

