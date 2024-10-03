import yaml
import numpy as np
import random 
from datetime import datetime
from typing import Union
from pathlib import Path

class Configs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, 'r') as f:
                config_dict = yaml.safe_load(f)
                        
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")
        
        self._set_attributes(config_dict) # set attributes recursively 
        
        if hasattr(self, 'experiment'):
            if not hasattr(self.experiment, 'name'):
                self.experiment.name = f"{self.data.source.name}_to_{self.data.target.name}_{self.dynamics.continuous.process}_{self.dynamics.discrete.process}_{self.model.name}"
                time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
                rnd = random.randint(0, 10000)
                self.experiment.name += f"_{time}_{rnd}"
                print('INFO: created experiment instance {}'.format(self.experiment.name)) 

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # create a sub-config object
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

    def save(self, path):
        """
        Saves the configuration parameters to a YAML file.
        """
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
# def import_model(config: Union[Configs, str]):

#     from cmb.dynamics.cfm import ConditionalFlowMatching, BatchOTCFM, BatchSBCFM
#     from cmb.dynamics.cmb import ConditionalMarkovBridge, BatchOTCMB, BatchSBCMB
#     from cmb.models.architectures.deep_nets import MLP, HybridMLP, ClassifierMLP
#     from cmb.models.architectures.epic import EPiC, HybridEPiC

#     if isinstance(config, str):
#         config = Configs(config)

#     model = locals()[config.model.name](config)
#     dynamics = locals()[config.dynamics.name](config)
#     print('      - model: {}'.format(config.model.name))
#     return config, model, dynamics, 
