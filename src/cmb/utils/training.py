
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class Train_Step(nn.Module): 
    """ Represents a training step.
    """
    def __init__(self):
        super(Train_Step, self).__init__()
        self.loss = 0
        self.epoch = 0
        self.losses = []

    def update(self, 
               model, 
               loss_fn, 
               dataloader: DataLoader, 
               optimizer, 
               scheduler):
        
        self.loss = 0
        self.epoch += 1
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            loss_current = loss_fn(model, batch)
            loss_current.backward()
            optimizer.step()
            scheduler.step(epoch=self.epoch)
            self.loss += loss_current.detach().cpu().numpy() / len(dataloader)
        self.losses.append(self.loss) 

class Validation_Step(nn.Module):
    """ Represents a validation step.
    """
    def __init__(self):
        super(Validation_Step, self).__init__()
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.losses = []
        
    @torch.no_grad()
    def update(self, 
               model, 
               loss_fn, 
               dataloader: DataLoader):
        
        self.epoch += 1
        self.loss = 0
        self.validate = bool(dataloader)
        if self.validate:
            model.eval()
            for batch in dataloader:
                loss_current = loss_fn(model, batch)
                self.loss += loss_current.detach().cpu().numpy() / len(dataloader)
            self.losses.append(self.loss) 

    @torch.no_grad()
    def checkpoint(self, min_epochs, early_stopping=None):
        terminate = False
        improved = False
        if self.validate:
            if self.loss < self.loss_min:
                self.loss_min = self.loss
                self.patience = 0
                improved = True 
            else: self.patience += 1 if self.epoch > min_epochs else 0
            if self.patience >= early_stopping: terminate = True
        return terminate, improved
    
class Test_Step(nn.Module):
    """ TODO Represents a test step.
    """
    def __init__(self):
        super(Test_Step, self).__init__()
        pass