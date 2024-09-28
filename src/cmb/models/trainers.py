import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.nn import DataParallel
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from typing import Union

from cmb.configs.experiments import Configs, import_model
from cmb.datasets.dataloader import DataloaderModule
from cmb.models.utils import Train_Step, Validation_Step, Optimizer, Scheduler, Logger

class GenerativeDynamicsModule:
    """
    Trainer for dynamic generative models. e.g. CFM
    
    Attributes:
    - dynamics: The model dynamics to train.
    - dataloader: DataLoader providing training and optionally validation data.
    - config: Configuration dataclass containing training configurations.
    """

    def __init__(self, 
                 config: Union[Configs, str]=None, 
                 dataclass=None):

        self.dataclass = dataclass
        self.config, self.model, self.dynamics = import_model(config=config)
        self.model = self.model.to(torch.device(self.config.train.device))
        self.workdir = Path(self.config.experiment.workdir) / Path(self.config.data.dataset) / Path(self.config.experiment.name)

    def train(self):

        #...train config:
        train = Train_Step()
        valid = Validation_Step()
        optimizer = Optimizer(self.config.train.optimizer)(self.model.parameters())
        scheduler = Scheduler(self.config.train.scheduler)(optimizer)
        early_stopping = self.config.train.epochs if self.config.train.early_stopping is None else self.config.train.early_stopping
        min_epochs = 0 if self.config.train.min_epochs is None else self.config.train.min_epochs
        print_epochs = 1 if self.config.train.print_epochs is None else self.config.train.print_epochs

        #...logging:
        
        os.makedirs(self.workdir, exist_ok=True)
        self.config.save(self.workdir / 'config.yaml')
        self.logger = Logger(self.workdir / 'training.log')
        self.logger.logfile.info("INFO: Training configurations:")
        self.config.log_config(self.logger)  # Log the nested configurations
        self.logger.logfile_and_console('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.logger.logfile.info(f"INFO: model architecture:\n{self.model}")
        self.logger.logfile_and_console("INFO: start training...")

        #...multi-gpu:
        if self.config.train.multi_gpu and torch.cuda.device_count() > 1:
            print("INFO: using ", torch.cuda.device_count(), "GPUs...")
            self.model = DataParallel(self.model)

        #...train loop:
            
        dataloader = DataloaderModule(self.config, self.dataclass)

        for epoch in tqdm(range(self.config.train.epochs), desc="epochs"):
            train.update(model=self.model, loss_fn=self.dynamics.loss, dataloader=dataloader.train, optimizer=optimizer) 
            valid.update(model=self.model, loss_fn=self.dynamics.loss, dataloader=dataloader.valid)
            TERMINATE, IMPROVED = valid.checkpoint(min_epochs=min_epochs, early_stopping=early_stopping)
            scheduler.step() 
            self._log_losses(train, valid, epoch, print_epochs)
            self._save_best_epoch_model(IMPROVED)
            
            if TERMINATE: 
                stop_message = "early stopping triggered! Reached maximum patience at {} epochs".format(epoch)
                self.logger.logfile_and_console(stop_message)
                break
            
        self._save_last_epoch_model()
        self._save_best_epoch_model(not bool(dataloader.valid)) # best = last epoch if there is no validation, needed as a placeholder for pipeline
        self.plot_loss(valid_loss=valid.losses, train_loss=train.losses)
        self.logger.close()
        
    def load(self, best_epoch_model: bool=True):
        if best_epoch_model:
            print('INFO: loading `best` epoch checkpoint from:') 
            print('  - {}'.format(self.workdir/'best_epoch_model.pth'))
            self.best_epoch_model = type(self.model)(self.config)
            self.best_epoch_model.load_state_dict(torch.load(self.workdir/'best_epoch_model.pth', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
        else:
            print('INFO: loading `last` epoch checkpoint from:') 
            print('  - {}'.format(self.workdir/'last_epoch_model.pth'))
            self.last_epoch_model = type(self.model)(self.config)
            self.last_epoch_model.load_state_dict(torch.load(self.workdir/'last_epoch_model.pth', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))

    @torch.no_grad()
    def generate(self, best_epoch_model: bool=True, **kwargs):
        assert hasattr(self, 'best_epoch_model') or hasattr(self, 'last_epoch_model'), "ERROR: No model checkpoint found. Please load a trained model first."
        time_steps = torch.linspace(0.0, 1.0 - self.config.pipeline.time_eps, self.config.pipeline.num_timesteps)
        model = self.best_epoch_model if best_epoch_model else self.last_epoch_model
        self.paths, self.jumps = self.dynamics.solver.simulate(model, time_steps=time_steps, **kwargs)
        
        if self.paths is not None and self.jumps is None: 
            sample = self.paths[-1]
        elif self.paths is None and self.jumps is not None: 
            sample = self.jumps[-1]
        elif self.paths is not None and self.jumps is not None: 
            sample = torch.cat([self.paths[-1], self.jumps[-1]], dim=-1)
        else: 
            raise ValueError("Both paths and jumps cannot be None simultaneously.")
        
        mask = kwargs.get('mask', None)
        self.sample = sample if mask is None else torch.cat([sample, mask], dim=-1)


    def _save_best_epoch_model(self, improved):
        if improved:
            torch.save(self.model.state_dict(), self.workdir/'best_epoch_model.pth')
            self.best_epoch_model = deepcopy(self.model)
        else: pass

    def _save_last_epoch_model(self):
        torch.save(self.model.state_dict(), self.workdir/'last_epoch_model.pth') 
        self.last_epoch_model = deepcopy(self.model)

    def _log_losses(self, train, valid, epoch, print_epochs=1):
        message = "\tEpoch: {}, train loss: {}, valid loss: {}  (min valid loss: {})".format(epoch, train.loss, valid.loss, valid.loss_min)
        self.logger.logfile.info(message)
        if epoch % print_epochs == 1:            
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
