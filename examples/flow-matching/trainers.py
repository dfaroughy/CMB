import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from torch.optim import Optimizer as TorchOptimizer
import inspect

from utils import DataloaderModule

class CFMTrainer:

    def __init__(self, config, dynamics, drift_model, dataclass):

        self.config = config
        self.workdir = Path(config.general.workdir) / Path(config.data.dataset) / Path(config.general.experiment_name)
        self.dynamics = dynamics
        self.model = drift_model
        self.dataloader = DataloaderModule(config, dataclass)

        #...train config:

        self.early_stopping = config.train.epochs if config.train.early_stopping is None else config.train.early_stopping
        self.min_epochs = 0 if config.train.min_epochs is None else config.train.min_epochs
        self.print_epochs = 1 if config.train.print_epochs is None else config.train.print_epochs

        #...logger & tensorboard:

        os.makedirs(self.workdir, exist_ok=True)
        self.logger = Logger(self.workdir/'training.log')
        self.logger.logfile.info("Training configurations:")
        self.config.log_config(self.logger)  # Log the nested configurations

        #...load model on device:

        self.model = self.model.to(torch.device(config.train.device))

    def train(self):
        train = Train_Step()
        valid = Validation_Step()
        optimizer = Optimizer(self.config.train.optimizer)(self.model.parameters())
        scheduler = Scheduler(self.config.train.scheduler)(optimizer)

        #...logging
        self.logger.logfile_and_console('number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.logger.logfile.info(f"Model architecture:\n{self.model}")
        self.logger.logfile_and_console("start training...")

        #...train

        if self.config.train.multi_gpu and torch.cuda.device_count() > 1:
            print("INFO: using ", torch.cuda.device_count(), "GPUs...")
            self.model = DataParallel(self.model)

        for epoch in tqdm(range(self.config.train.epochs), desc="epochs"):
            train.update(model=self.model, loss_fn=self.dynamics.loss, dataloader=self.dataloader.train, optimizer=optimizer) 
            valid.update(model=self.model, loss_fn=self.dynamics.loss, dataloader=self.dataloader.valid)
            TERMINATE, IMPROVED = valid.checkpoint(min_epochs=self.min_epochs, 
                                                   early_stopping=self.early_stopping)
            scheduler.step() 
            self._log_losses(train, valid, epoch)
            self._save_best_epoch_model(IMPROVED)
            
            if TERMINATE: 
                stop_message = "early stopping triggered! Reached maximum patience at {} epochs".format(epoch)
                self.logger.logfile_and_console(stop_message)
                break
            
        self._save_last_epoch_model()
        self._save_best_epoch_model(not bool(self.dataloader.valid)) # best = last epoch if there is no validation, needed as a placeholder for pipeline
        self.plot_loss(valid_loss=valid.losses, train_loss=train.losses)
        self.logger.close()
        
    def load(self, path: str=None, model: str=None):
        path = self.workdir if path is None else Path(path)
        if model is None:
            self.best_epoch_model = type(self.model)(self.config)
            self.last_epoch_model = type(self.model)(self.config)
            self.best_epoch_model.load_state_dict(torch.load(path/'best_epoch_model.pth', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
            self.last_epoch_model.load_state_dict(torch.load(path/'last_epoch_model.pth', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
        elif model == 'best':
            self.best_epoch_model = type(self.model)(self.config)
            self.best_epoch_model.load_state_dict(torch.load(path/'best_epoch_model.pth', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
        elif model == 'last':
            self.last_epoch_model = type(self.model)(self.config)
            self.last_epoch_model.load_state_dict(torch.load(path/'last_epoch_model.pth', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
        else: raise ValueError("which_model must be either 'best', 'last', or None")

    def _save_best_epoch_model(self, improved):
        if improved:
            torch.save(self.model.state_dict(), self.workdir/'best_epoch_model.pth')
            self.best_epoch_model = deepcopy(self.model)
        else: pass

    def _save_last_epoch_model(self):
        torch.save(self.model.state_dict(), self.workdir/'last_epoch_model.pth') 
        self.last_epoch_model = deepcopy(self.model)

    def _log_losses(self, train, valid, epoch):
        message = "\tEpoch: {}, train loss: {}, valid loss: {}  (min valid loss: {})".format(epoch, train.loss, valid.loss, valid.loss_min)
        self.logger.logfile.info(message)
        if epoch % self.print_epochs == 1:            
            self.plot_loss(valid_loss=valid.losses, train_loss=train.losses)
            self.logger.console.info(message)

    def plot_loss(self, valid_loss, train_loss):
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(range(len(valid_loss)), np.array(valid_loss), color='r', lw=1, linestyle='-', label='Validation')
        ax.plot(range(len(train_loss)), np.array(train_loss), color='b', lw=1, linestyle='--', label='Training', alpha=0.8)
        ax.set_xlabel("Epochs", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.set_title("Training & Validation Loss Over Epochs", fontsize=6)
        ax.set_yscale('log')
        ax.legend(fontsize=6)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        plt.savefig(self.workdir / 'losses.png')
        plt.close()


#----------------------------------------------
# utils for trainers           
#----------------------------------------------
        

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, fields


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


class Optimizer:
    """
    Custom optimizer class with support for gradient clipping.
    
    Attributes:
    - config: Configuration object containing optimizer configurations.
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, parameters):
        config_dict = self.config.to_dict()
        optimizer_name = self._get_optimizer_name(config_dict)
        optimizer_cls = self._get_optimizer_class(optimizer_name)
        valid_args = self._get_valid_args(optimizer_cls)
        optimizer_args = {k: v for k, v in config_dict.items() if k in valid_args}
        self.gradient_clip = config_dict.get('gradient_clip', None)
        optimizer = optimizer_cls(parameters, **optimizer_args)
        optimizer = self._wrap_optimizer_step(optimizer)

        return optimizer

    def _wrap_optimizer_step(self, optimizer):
        original_step = optimizer.step

        def step_with_clipping(closure=None):
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.gradient_clip)
            original_step(closure)
        optimizer.step = step_with_clipping
        return optimizer

    def _get_optimizer_name(self, config_dict):
        optimizer_names = [cls_name for cls_name in dir(torch.optim) if isinstance(getattr(torch.optim, cls_name), type)]
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

