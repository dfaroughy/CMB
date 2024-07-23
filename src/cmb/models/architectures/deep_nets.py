import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


from cmb.models.architectures.utils import (fc_block, 
                                            get_activation_function, 
                                            transformer_timestep_embedding,
                                            timestep_sinusoidal_embedding,
                                            GaussianFourierProjection)

#...Multi-Layer Perceptron architecture:



class MLP(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.device = configs.DEVICE
        self.define_deep_models(configs)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, configs):
        self.dim_input = configs.DIM_INPUT
        self.dim_output = configs.DIM_INPUT
        self.dim_hidden = configs.DIM_HIDDEN
        self.dim_context = configs.DIM_CONTEXT
        self.time_embedding = configs.TIME_EMBEDDING 
        self.dim_time_emb = configs.DIM_TIME_EMB if configs.DIM_TIME_EMB is not None else 1
        self.num_layers = configs.NUM_LAYERS
        self.dropout = configs.DROPOUT
        self.act_fn = get_activation_function(configs.ACTIVATION)

        self.layers = fc_block(dim_input=self.dim_input + self.dim_context + self.dim_time_emb, 
                               dim_output=self.dim_output, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=True)

    def forward(self, t, x, context=None, mask=None):
        x = x.to(self.device)
        t = t.to(self.device)
        context = context.to(self.device) if context is not None else None
        
        if self.time_embedding == 'sinusoidal': t_emb = timestep_sinusoidal_embedding(t, self.dim_time_emb, max_period=10000)
        elif self.time_embedding == 'gaussian': t_emb = GaussianFourierProjection(self.dim_time_emb, device=x.device)(t)
        else: t_emb = transformer_timestep_embedding(t.squeeze(1), embedding_dim=self.dim_time_emb) if t is not None else t
        x = torch.concat([x, context, t_emb.squeeze()], dim=1) if context is not None else torch.concat([x, t_emb.squeeze()], dim=1) 
        return self.layers(x)

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
#...ResNet architecture:

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, num_layers_per_block=2):
        super(ResidualBlock, self).__init__()

        layers = []
        for _ in range(num_layers_per_block):
            layers.extend([torch.nn.Linear(dim, dim), torch.nn.SELU()])
        self.block = torch.nn.Sequential(*layers[:-1])

    def forward(self, x):
        return x + self.block(x)
    
class _ResNet(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, num_blocks=3, num_layers_per_block=2):
        super(_ResNet, self).__init__()

        if out_dim is None:
            out_dim = dim

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU()
        )
        self.blocks = torch.nn.ModuleList([ResidualBlock(w, num_layers_per_block) for _ in range(num_blocks)])
        self.output_layer = torch.nn.Linear(w, out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class ResNet(nn.Module):
    ''' Wrapper class for the ResNet architecture
    '''
    def __init__(self, configs):
        super(ResNet, self).__init__()
        self.device = configs.DEVICE
        self.resnet = _ResNet(dim=configs.dim_input, 
                            out_dim=None,
                            w=configs.dim_hidden, 
                            num_blocks=configs.num_blocks,
                            num_layers_per_block=configs.num_block_layers,
                            time_varying=True)
                
    def forward(self, t, x, context=None, mask=None, sampling=False):
        x = torch.cat([x, t], dim=-1)
        x = x.to(self.device)
        self.resnet = self.resnet.to(self.device)
        return self.resnet.forward(x)
