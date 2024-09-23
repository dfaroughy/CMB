import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax
import numpy as np

from cmb.models.architectures.utils import (fc_blocks, 
                                            kan_blocks,
                                            get_activation_function, 
                                            InputEmbeddings,
                                            KANLinear, KAN,
                                            SinusoidalPositionalEncoding,
                                            GaussianFourierFeatures)

class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.define_deep_models(config)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, config):
        self.dim_continuous = config.dim_continuous
        self.dim_discrete = config.dim_discrete
        self.dim_context = config.dim_context
        self.dim_hidden = config.dim_hidden
        self.dim_time_emb = config.dim_time_emb 
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.act_fn = get_activation_function(config.activation)
        self.use_batch_norm = config.use_batch_norm

        #...Time embedding:

        if hasattr(config, 'time_embedding'):
            if config.time_embedding == 'SinusoidalPositionalEncoding': self.time_embedding = SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000)
            elif config.time_embedding == 'GaussianFourierFeatures': self.time_embedding = GaussianFourierFeatures(self.dim_time_emb, scale=0.5)
            elif config.time_embedding == 'KANLinear': self.time_embedding = KANLinear(1, self.dim_time_emb)
            elif config.time_embedding == 'Linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)  
            else: raise NotImplementedError                                                              
        else:
            self.dim_time_emb = 1

        #...Discrete embedding:
            
        if self.dim_discrete > 0: 
            self.dim_discrete_emb = config.dim_discrete_emb
            self.discrete_embedding = nn.Embedding(config.vocab_size, self.dim_discrete_emb)
        else:
            self.dim_discrete_emb = 0


        #...Context embedding:
            
        if hasattr(config, 'context_embedding') and self.dim_context > 0: 
            self.dim_context_emb = config.dim_context_emb
            if config.context_embedding == 'Embedding': self.context_embedding = nn.Embedding(config.vocab_size, self.dim_context_emb)
            elif config.context_embedding == 'Linear': self.time_embedding = nn.Linear(self.dim_context, self.dim_context_emb)  
            else: raise NotImplementedError                                                              
        else:
            self.dim_context_emb = self.dim_context if self.dim_context > 0 else 0
        
        #...MLP layers:
                        
        self.layers = fc_block(dim_input=self.dim_continuous + self.dim_context_emb + self.dim_discrete_emb + self.dim_time_emb, 
                               dim_output=self.dim_continuous + config.dim_discrete * config.vocab_size, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=self.use_batch_norm)


    def forward(self, t, x=None, k=None, context=None, mask=None):
        
        t = t.to(self.device)
        t = self.time_embedding(t) if hasattr(self, 'time_embedding') else t
        features = [t]

        if x is not None:
            x = x.to(self.device)
            features.append(x)

        if k is not None:
            k = k.to(self.device)
            k = self.discrete_embedding(k) if hasattr(self, 'context_embedding') else context
            features.append(context)

        if context is not None:
            context = context.to(self.device)
            context = self.context_embedding(context) if hasattr(self, 'context_embedding') else context
            features.append(context)

        if mask is not None:
            mask = mask.to(self.device)

        h = torch.concat(features, dim=1) 
        h = self.layers(h)
        continuous = h[:, :self.dim_continuous]   # vector field
        discrete = h[:, self.dim_continuous:]
        discrete = discrete.reshape(k.size(0), self.dim_input, self.vocab_size)   # rate logits
        return continuous, discrete

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.define_deep_models(config)
        self.to(self.device)

    def define_deep_models(self, config):
        self.dim_input = config.dim_continuous
        self.dim_output = config.dim_continuous
        self.dim_hidden = config.dim_hidden
        self.dim_context = config.dim_context
        self.dim_continuous_emb  = config.dim_continuous_emb 
        self.dim_time_emb = config.dim_time_emb 
        self.dim_context_emb = config.dim_context_emb
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.act_fn = get_activation_function(config.activation)

        if config.time_embedding == 'SinusoidalPositionalEncoding': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.time_embedding == 'GaussianFourierFeatures': self.time_embedding = nn.Sequential(GaussianFourierFeatures(self.dim_time_emb, scale=0.5),  nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.time_embedding == 'KANLinear': self.time_embedding = nn.Sequential(KANLinear(1, self.dim_time_emb), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.time_embedding == 'Linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)                                                                
        elif config.time_embedding is None: self.dim_time_emb = 1
        else: raise NotImplementedError

        self.kan = kan_block(dim_input=self.dim_input + self.dim_context + self.dim_time_emb, 
                            dim_output=self.dim_output, 
                            dim_hidden=self.dim_hidden, 
                            num_layers=self.num_layers, 
                            dropout=self.dropout, 
                            use_batch_norm=True)

    def forward(self, t, x, k=None, context=None):
        x = x.to(self.device)
        t = t.to(self.device)
        context = context.to(self.device) if context is not None else None
        t_emb = self.time_embedding(t) if hasattr(self, 'time_embedding') else t
        h = torch.concat([x, context, t_emb], dim=1) if context is not None else torch.concat([x, t_emb], dim=1) 
        return self.kan(h)






