import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, fields

#----------------------------------------------
# utils for trainers           
#----------------------------------------------

class Train_Step(nn.Module): 

    """ Represents a training step.
    """

    def __init__(self):
        super(Train_Step, self).__init__()
        self.loss = 0
        self.epoch = 0
        self.losses = []

    def update(self, model, loss_fn, dataloader: DataLoader, optimizer):
        self.loss = 0
        self.epoch += 1
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            loss_current = loss_fn(model, batch)
            loss_current.backward()
            optimizer.step()  
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
    def update(self, model, loss_fn, dataloader: DataLoader):
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

import torch
from torch.optim import Optimizer as TorchOptimizer
import inspect

class Optimizer:
    """
    Custom optimizer class with support for gradient clipping.
    
    Attributes:
    - config: Configuration object containing optimizer configurations.
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, parameters):
        # Convert the config to a dictionary
        config_dict = self.config.to_dict()

        # Get the optimizer name dynamically
        optimizer_name = self._get_optimizer_name(config_dict)

        # Get the optimizer class from torch.optim
        optimizer_cls = self._get_optimizer_class(optimizer_name)

        # Get valid arguments for the optimizer constructor
        valid_args = self._get_valid_args(optimizer_cls)

        # Filter config_dict to include only valid arguments
        optimizer_args = {k: v for k, v in config_dict.items() if k in valid_args}

        # Extract gradient clipping value if present
        self.gradient_clip = config_dict.get('gradient_clip', None)

        # Create the optimizer instance
        optimizer = optimizer_cls(parameters, **optimizer_args)

        # Override optimizer.step() to include gradient clipping
        optimizer = self._wrap_optimizer_step(optimizer)

        return optimizer

    def _wrap_optimizer_step(self, optimizer):
        # Original optimizer step function
        original_step = optimizer.step

        def step_with_clipping(closure=None):
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.gradient_clip)
            original_step(closure)
        optimizer.step = step_with_clipping
        return optimizer

    def _get_optimizer_name(self, config_dict):
        # Get a list of all optimizer class names in torch.optim
        optimizer_names = [cls_name for cls_name in dir(torch.optim) if isinstance(getattr(torch.optim, cls_name), type)]
        # Find keys in config_dict that are not valid optimizer arguments
        possible_names = set(config_dict.keys()) - set(self._get_all_optimizer_args())

        for key in possible_names:
            value = config_dict[key]
            if isinstance(value, str) and value in optimizer_names:
                return value
            elif key in optimizer_names:
                return key
        raise ValueError("Optimizer name not found in configuration. Please specify a valid optimizer name.")

    def _get_optimizer_class(self, optimizer_name):
        if hasattr(torch.optim, optimizer_name):
            return getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_valid_args(self, optimizer_cls):
        # Get the signature of the optimizer class
        signature = inspect.signature(optimizer_cls.__init__)
        # Exclude 'self' and 'params' from the parameters
        valid_args = [p.name for p in signature.parameters.values() if p.name not in ['self', 'params']]
        return valid_args

    def _get_all_optimizer_args(self):
        # Get all valid arguments from all optimizers in torch.optim
        all_args = set()
        for attr_name in dir(torch.optim):
            attr = getattr(torch.optim, attr_name)
            if inspect.isclass(attr) and issubclass(attr, TorchOptimizer):
                args = self._get_valid_args(attr)
                all_args.update(args)
        return all_args
    

# class Optimizer:

#     """
#     Custom optimizer class with support for gradient clipping.
    
#     Attributes:
#     - configs: Configuration dataclass containing optimizer configurations.
#     """

#     def __init__(self, config: dataclass):
#         self.config = config

#     def get_optimizer(self, parameters):

#         args = self.config.to_dict()

#         if self.config.name == 'Adam':
#             return torch.optim.Adam(parameters, **args)
#         elif self.config.name == 'AdamW':
#             return torch.optim.AdamW(parameters, **args)
#         else:
#             raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
#     def clip_gradients(self, optimizer):
#         if self.config.gradient_clip: torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.config.gradient_clip)

#     def __call__(self, parameters):
#         optimizer = self.get_optimizer(parameters)
#         #...override the optimizer.step() to include gradient clipping
#         original_step = optimizer.step

#         def step_with_clipping(closure=None):
#             self.clip_gradients(optimizer)
#             original_step(closure)          
#         optimizer.step = step_with_clipping
#         return optimizer

class Scheduler:

    """
    Custom scheduler class to adjust the learning rate during training.
    
    Attributes:
    - configs: Configuration dataclass containing scheduler configurations.
    """

    def __init__(self, config: dataclass):
        self.config = config 

    def get_scheduler(self, optimizer):
        args = self.config.to_dict()
        if self.config.name == 'CosineAnnealingLR': return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **args)
        elif self.config.name == 'StepLR': return torch.optim.lr_scheduler.StepLR(optimizer, **args)
        elif self.config.name == 'ExponentialLR': return torch.optim.lr_scheduler.ExponentialLR(optimizer, **args)
        elif self.config.name is None: return NoScheduler(optimizer)
        else: raise ValueError(f"Unsupported scheduler: {self.scheduler}")

    def __call__(self, optimizer):
        return self.get_scheduler(optimizer)

class NoScheduler:
    def __init__(self, optimizer): pass
    def step(self): pass    


class Logger:
    ''' Logging handler for training and validation.
    '''
    def __init__(self, path: Path):
        self.path = path
        self.fh = None  
        self.ch = None 
        self._training_loggers()

    def _training_loggers(self):
        
        self.logfile = logging.getLogger('file_logger')
        self.logfile.setLevel(logging.INFO)
        self.fh = logging.FileHandler(self.path) 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.logfile.addHandler(self.fh)
        self.logfile.propagate = False 
        
        self.console = logging.getLogger('console_logger')
        self.console.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()  
        ch_formatter = logging.Formatter('%(message)s') 
        self.ch.setFormatter(ch_formatter)
        self.console.addHandler(self.ch)
        self.console.propagate = False 

    def logfile_and_console(self, message):
        self.logfile.info(message)
        self.console.info(message)

    def close(self):
        if self.fh:
            self.fh.close()
            self.logfile.removeHandler(self.fh)
        if self.ch:
            self.ch.close()
            self.console.removeHandler(self.ch)

