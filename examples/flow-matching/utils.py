#############################
# Configs helper functions
#############################

import yaml
import numpy as np
from datetime import datetime

class Configs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            # If config_source is a file path, load the YAML file
            with open(config_source, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            # If config_source is a dict, use it directly
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")
        
        # Recursively set attributes
        self._set_attributes(config_dict)
        
        if hasattr(self, 'data'):
            if hasattr(self.data, 'source') and hasattr(self.data, 'target'):
                self.general.experiment_name = f"{self.data.source.name}_to_{self.data.target.name}_{self.dynamics.name}_{self.model.name}"
                time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
                self.general.experiment_name += f"_{time}"

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Create a sub-config object
                sub_config = Configs(value)
                setattr(self, key, sub_config)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts the Configs object into a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configs):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def print(self):
        """
        Prints the configuration parameters in a structured format.
        """
        config_dict = self.to_dict()
        self._print_dict(config_dict)

    def _print_dict(self, config_dict, indent=0):
        """
        Helper method to recursively print the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = ' ' * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    def log_config(self, logger):
        """
        Logs the configuration parameters using the provided logger.
        """
        config_dict = self.to_dict()
        self._log_dict(config_dict, logger)

    def _log_dict(self, config_dict, logger, indent=0):
        """
        Helper method to recursively log the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = ' ' * indent
            if isinstance(value, dict):
                logger.logfile.info(f"{prefix}{key}:")
                self._log_dict(value, logger, indent + 4)
            else:
                logger.logfile.info(f"{prefix}{key}: {value}")


###############################
# Datamodules helper functions
###############################
                
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from collections import namedtuple


class DataloaderModule:
    def __init__(self, 
                 config,
                 dataclass, 
                 batch_size: int=None, 
                 data_split_frac: tuple=None): 
        self.dataclass = dataclass
        self.config = config       
        self.dataset = DataSetModule(dataclass) 
        self.data_split = self.config.train.data_split_frac if data_split_frac is None else data_split_frac
        self.batch_size = self.config.train.batch_size if batch_size is None else batch_size
        self.dataloader()

    def train_val_test_split(self, shuffle=False):
        assert np.abs(1.0 - sum(self.data_split)) < 1e-3, "Split fractions do not sum to 1!"
        total_size = len(self.dataset)
        train_size = int(total_size * self.data_split[0])
        valid_size = int(total_size * self.data_split[1])

        #...define splitting indices

        idx = torch.randperm(total_size) if shuffle else torch.arange(total_size)
        idx_train = idx[:train_size].tolist()
        idx_valid = idx[train_size : train_size + valid_size].tolist()
        idx_test = idx[train_size + valid_size :].tolist()
        
        #...Create Subset for each split

        train_set = Subset(self.dataset, idx_train)
        valid_set = Subset(self.dataset, idx_valid) if valid_size > 0 else None
        test_set = Subset(self.dataset, idx_test) if self.data_split[2] > 0 else None

        return train_set, valid_set, test_set


    def dataloader(self):

        print("INFO: building dataloaders...")
        print("INFO: train/val/test split ratios: {}/{}/{}".format(self.data_split[0], self.data_split[1], self.data_split[2]))
        
        train, valid, test = self.train_val_test_split(shuffle=True)
        self.train = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
        self.valid = DataLoader(dataset=valid,  batch_size=self.batch_size, shuffle=False) if valid is not None else None
        self.test = DataLoader(dataset=test,  batch_size=self.batch_size, shuffle=True) if test is not None else None

        print('INFO: train size: {}, validation size: {}, testing sizes: {}'.format(len(self.train.dataset),  # type: ignore
                                                                                    len(self.valid.dataset if valid is not None else []),  # type: ignore
                                                                                    len(self.test.dataset if test is not None else []))) # type: ignore

class DataSetModule(Dataset):
    def __init__(self, data):
        self.data = data

        self.attributes=[]

        if hasattr(self.data.source, 'particles'): 
            self.attributes.append('source')
            self.source = self.data.source.particles

        if hasattr(self.data.target, 'particles'): 
            self.attributes.append('target')
            self.target = self.data.target.particles

        if hasattr(self.data.target, 'context'): 
            self.attributes.append('context')
            self.context = self.data.target.context

        if hasattr(self.data.target, 'mask'): 
            self.attributes.append('mask')
            self.mask = self.data.target.mask

        self.databatch = namedtuple('databatch', self.attributes)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attributes])

    def __len__(self):
        return len(self.data.target)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


#############################
# NN helper functions
#############################

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """ Positional encoding with log-linear spaced frequencies for each dimension        
    """
    def __init__(self, dim, max_period=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp( -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.squeeze()
    

#############################
# Optimal transport 
#############################

import warnings
import numpy as np
import ot as pot
import torch

class OTPlanSampler:

    def __init__(self, reg: float = 0.05, reg_m: float = 1.0, normalize_cost: bool = False, warn: bool = True):
        self.ot_fn = pot.emd
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=False):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=False):
        pi = self.get_map(x0, x1)
        self.i, self.j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[self.i], x1[self.j]

