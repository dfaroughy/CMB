import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.nn import DataParallel
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from typing import Union

from cmb.utils.training import Train_Step, Validation_Step
from cmb.utils.loggers import Logger
from cmb.configs.registered_optimizers import optimizers, schedulers
from cmb.configs.registered_models import models
from cmb.configs.registered_generative_dynamics import dynamics
from cmb.configs.utils import Configs
from cmb.datasets.dataloader import DataloaderModule

class ClassifierMetric: 
    """ Trainer for dynamic generative models
    """
    def __init__(self, 
                 config: Union[Configs, str]=None, 
                 dataclass=None,
                 device=None):

        if isinstance(config, str): 
            config = Configs(config)                                              # get configs form yaml
            if device is not None: config.train.device = device                   # override config device if provided
        
        self.config = config 
        self.dataclass = dataclass
        self.dynamics = dynamics.get(self.config.dynamics.name)(self.config)           # get generative dynamical model from config
        self.model = models.get(self.config.model.name)(self.config)                   # get NN from configs
        self.model = self.model.to(torch.device(self.config.train.device))
        self.workdir = Path(self.config.experiment.workdir) / Path(self.config.experiment.run_name) / Path(self.config.experiment.name)
        
    def train(self):
        #...train config:
        train = Train_Step()
        valid = Validation_Step()
        optimizer = optimizers.get(self.config.train.optimizer.name)(self.model.parameters(), **self.config.train.optimizer.params.to_dict())
        scheduler = schedulers.get(self.config.train.scheduler.name)(optimizer=optimizer, **self.config.train.scheduler.params.to_dict())
        early_stopping = self.config.train.epochs if self.config.train.early_stopping is None else self.config.train.early_stopping
        min_epochs = 0 if self.config.train.min_epochs is None else self.config.train.min_epochs
        print_epochs = 1 if self.config.train.print_epochs is None else self.config.train.print_epochs

        #...multi-gpu:
        if self.config.train.multi_gpu and torch.cuda.device_count() > 1:
            print("INFO: using ", torch.cuda.device_count(), "GPUs...")
            self.model = DataParallel(self.model)

        #...preprocess data:
        if hasattr(self.config.data, 'preprocess'):
            print('INFO: Preprocessing data...')
            print('    - continuous data: {}'.format(self.config.data.preprocess.continuous))
            print('    - discrete data: {}'.format(self.config.data.preprocess.discrete))

            self.dataclass.source.preprocess(output_continuous=self.config.data.preprocess.continuous, 
                                             output_discrete=self.config.data.preprocess.discrete)
            
            self.dataclass.target.preprocess(output_continuous=self.config.data.preprocess.continuous, 
                                             output_discrete=self.config.data.preprocess.discrete)

            self.config.data.source.train.stats = self.dataclass.source.stats
            self.config.data.target.train.stats = self.dataclass.target.stats
        
        #...logging:
        os.makedirs(self.workdir, exist_ok=True)
        self.config.save(self.workdir / 'config.yaml')
        self.logger = Logger(self.workdir / 'training.log')
        self.logger.logfile.info("INFO: Training configurations:")
        self.config.log_config(self.logger)  # Log the nested configurations
        self.logger.logfile_and_console('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.logger.logfile.info(f"INFO: model architecture:\n{self.model}")
        self.logger.logfile_and_console("INFO: start training...")

        #...dataloader:
        dataloader = DataloaderModule(self.config, self.dataclass)

        #...train loop:
        for epoch in tqdm(range(self.config.train.epochs), desc="epochs"):
            train.update(model=self.model, 
                         loss_fn=self.dynamics.loss, 
                         dataloader=dataloader.train, 
                         optimizer=optimizer, 
                         scheduler=scheduler,
                         gradient_clip=self.config.train.optimizer.gradient_clip) 
            valid.update(model=self.model, 
                         loss_fn=self.dynamics.loss, 
                         dataloader=dataloader.valid)
            
            TERMINATE, IMPROVED = valid.checkpoint(min_epochs=min_epochs, early_stopping=early_stopping)
            self._log_losses(train, valid, epoch, print_epochs)
            self._save_best_epoch_ckpt(IMPROVED)
            self._save_last_epoch_ckpt()
            
            if TERMINATE: 
                stop_message = "early stopping triggered! Reached maximum patience at {} epochs".format(epoch)
                self.logger.logfile_and_console(stop_message)
                break
            
        self.val_losses = valid.losses            
        self._save_last_epoch_ckpt()
        self._save_best_epoch_ckpt(not bool(dataloader.valid)) # best = last epoch if there is no validation, needed as a placeholder for pipeline
        self._plot_loss(valid_loss=valid.losses, train_loss=train.losses)
        self.logger.close()
        
    def load(self, checkpoint: str='best'):
        if checkpoint=='best':
            print('INFO: loading `best` epoch checkpoint on {} from:'.format(self.config.train.device)) 
            print('  - {}'.format(self.workdir/'best_epoch.ckpt'))
            self.best_epoch_ckpt = type(self.model)(self.config)
            self.best_epoch_ckpt.load_state_dict(torch.load(self.workdir/'best_epoch.ckpt', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
        elif checkpoint=='last':
            print('INFO: loading `last` epoch checkpoint on {} from:'.format(self.config.train.device)) 
            print('  - {}'.format(self.workdir/'last_epoch.ckpt'))
            self.last_epoch_ckpt = type(self.model)(self.config)
            self.last_epoch_ckpt.load_state_dict(torch.load(self.workdir/'last_epoch.ckpt', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))
        else:
            print('INFO: loading `{}` checkpoint on {} from:'.format(checkpoint, self.config.train.device)) 
            print('  - {}'.format(self.workdir/f'{checkpoint}.ckpt'))
            self.checkpoint = type(self.model)(self.config)
            self.checkpoint.load_state_dict(torch.load(self.workdir/f'{checkpoint}.ckpt', map_location=(torch.device('cpu') if self.config.train.device=='cpu' else None)))



    @torch.no_grad()
    def generate(self, dataclass=None, output_history=False, save_to=None, **kwargs):

        print('INFO: generating samples...') 
        if hasattr(self, 'best_epoch_ckpt'):  model = self.best_epoch_ckpt 
        elif hasattr(self, 'last_epoch_ckpt'):  model = self.last_epoch_ckpt
        else:  model = self.checkpoint

        time_steps = torch.linspace(0.0, 1.0 - self.config.pipeline.time_eps, self.config.pipeline.num_timesteps)
        out_continuous, out_discrete = self.dynamics.solver.simulate(model, 
                                                                     time_steps=time_steps, 
                                                                     output_history=output_history,  
                                                                     **kwargs)
        if out_continuous is not None and out_discrete is None: 
            sample = out_continuous[-1] if output_history else out_continuous
            self.trajectories = out_continuous if output_history else None

        elif out_continuous is None and out_discrete is not None: 
            sample = out_discrete[-1] if output_history else out_discrete
            self.jumps = out_discrete if output_history else None

        elif out_continuous is not None and out_discrete is not None: 
            sample = torch.cat([out_continuous[-1], out_discrete[-1]], dim=-1) if output_history else torch.cat([out_continuous, out_discrete], dim=-1)
            self.trajectories = out_continuous if output_history else None
            self.jumps = out_discrete if output_history else None
        else: 
            raise ValueError("Both trajectories and jumps cannot be `None` simultaneously.")
        
        mask = kwargs.get('mask', None)
        self.sample = sample if mask is None else torch.cat([sample, mask], dim=-1)

        # save sample to file, where sample is a torch tensor:
        if save_to is not None:
            torch.save(self.sample, self.workdir / f"{save_to}")   

        if dataclass is not None:
            self.sample = dataclass(self.sample)
            self.sample.stats = self.config.data.target.train.stats 
            if not isinstance(self.sample.stats, dict):
                self.sample.stats = self.sample.stats.to_dict()
            self.sample.postprocess(input_continuous=self.config.data.preprocess.continuous, 
                                    input_discrete=self.config.data.preprocess.discrete)

    def _save_best_epoch_ckpt(self, improved):
        if improved:
            torch.save(self.model.state_dict(), self.workdir/f'best_epoch.ckpt')
            self.best_epoch_ckpt = deepcopy(self.model)
        else: pass

    def _save_last_epoch_ckpt(self):
        torch.save(self.model.state_dict(), self.workdir/'last_epoch.ckpt') 
        self.last_epoch_ckpt = deepcopy(self.model)

    def _log_losses(self, train, valid, epoch, print_epochs=1):
        message = "\tEpoch: {}, train loss: {}, valid loss: {}  (min valid loss: {})".format(epoch, train.loss, valid.loss, valid.loss_min)
        self.logger.logfile.info(message)
        if epoch % print_epochs == 1:            
            self._plot_loss(valid_loss=valid.losses, train_loss=train.losses)
            self.logger.console.info(message)

    def _plot_loss(self, valid_loss, train_loss):
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(range(len(valid_loss)), np.array(valid_loss), color='darkred', lw=0.75, linestyle='--', label='val')
        ax.plot(range(len(train_loss)), np.array(train_loss), color='darkblue', lw=0.75, linestyle='-', label='train')
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
