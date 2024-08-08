from dataclasses import dataclass, field
from cmb.models.configs import Training_Config
from typing import List

""" Default configurations for models.
"""

@dataclass
class MLP_Config(Training_Config):
    MODEL : str = 'MLP'
    DIM_HIDDEN : int = 128  
    TIME_EMBEDDING_TYPE : str = 'sinusoidal'
    DIM_TIME_EMB : int = 16
    NUM_LAYERS : int = 3
    DROPOUT : float = 0.0
    ACTIVATION : str = 'ReLU'


@dataclass
class StateClassifier_Config(Training_Config):
    MODEL : str = 'MLP'
    DIM_HIDDEN : int = 128  
    TIME_EMBEDDING_TYPE : str = 'sinusoidal'
    DIM_TIME_EMB : int = 16
    DIM_STATE_EMB : int = 8
    NUM_LAYERS : int = 3
    DROPOUT : float = 0.0
    ACTIVATION : str = 'ReLU'

@dataclass
class Unet28x28_Config(Training_Config):
    MODEL : str = 'Unet28x28'
    DIM_HIDDEN : int = 64 
    DIM_TIME_EMB : int = 32
    ACTIVATION : str = 'GELU'
    DROPOUT : float = 0.1

@dataclass
class Unet32x32_Config(Training_Config):
    MODEL : str = 'Unet32x32'
    NUM_CHANNELS : int = 3
    CHANNEL_MULT : List[float] = field(default_factory = lambda : [1, 2, 2, 2])
    NUM_RES_BLOCKS : int = 2
    NUM_HEADS : int = 4
    DIM_HIDDEN : int = 64 
    ATTENTION_RESOLUTIONS : str = "16"
    DROPOUT : float = 0.1
