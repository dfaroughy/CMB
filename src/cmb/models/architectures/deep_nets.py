import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


from cmb.models.architectures.utils import (fc_block, 
                                            kan_block,
                                            get_activation_function, 
                                            KANLinear, KAN,
                                            SinusoidalPositionalEncoding,
                                            GaussianFourierFeatures)

#...Multi-Layer Perceptron architecture:

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.define_deep_models(config)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, config):
        self.dim_input = config.DIM_INPUT
        self.dim_output = config.DIM_INPUT
        self.dim_hidden = config.DIM_HIDDEN
        self.dim_context = config.DIM_CONTEXT
        self.dim_feat_emb = config.DIM_FEAT_EMB if config.FEAT_ENCODING_TYPE is not None else config.DIM_INPUT
        self.dim_time_emb = config.DIM_TIME_EMB if config.TIME_ENCODING_TYPE is not None else 1
        self.dim_context_emb = config.DIM_CONTEXT_EMB if config.CONTEXT_ENCODING_TYPE is not None else config.DIM_CONTEXT
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT
        self.act_fn = get_activation_function(config.ACTIVATION)

        if config.FEAT_ENCODING_TYPE == 'linear': self.feature_embedding = nn.Linear(self.dim_input, self.dim_feat_emb)   
        elif config.TIME_ENCODING_TYPE == 'kolmogorov-arnold': self.feature_embedding = nn.Sequential(KANLinear(self.dim_input, self.dim_feat_emb), nn.Linear(self.dim_feat_emb, self.dim_feat_emb))                     
        elif config.FEAT_ENCODING_TYPE is None: self.feature_embedding = nn.Identity()
        else: raise NotImplementedError 

        if config.TIME_ENCODING_TYPE == 'sinusoidal': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'randomfourier': self.time_embedding = nn.Sequential(GaussianFourierFeatures(self.dim_time_emb, scale=0.5),  nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'kolmogorov-arnold': self.time_embedding = nn.Sequential(KANLinear(1, self.dim_time_emb), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)                                                                
        elif config.TIME_ENCODING_TYPE is None: self.time_embedding = nn.Identity()
        else: raise NotImplementedError

        if config.FEAT_ENCODING_TYPE == 'linear': self.context_embedding = nn.Linear(self.dim_context, self.dim_context_emb)   
        elif config.TIME_ENCODING_TYPE == 'kolmogorov-arnold': self.context_embedding = nn.Sequential(KANLinear(self.dim_context, self.dim_context_emb), nn.Linear(self.dim_context_emb, self.dim_context_emb))                     
        elif config.FEAT_ENCODING_TYPE is None: self.context_embedding = nn.Identity()
        else: raise NotImplementedError 
        
        self.layers = fc_block(dim_input=self.dim_feat_emb + self.dim_context_emb + self.dim_time_emb, 
                               dim_output=self.dim_output, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=True)

    def forward(self, t, x, s=None, context=None):
        x = x.to(self.device)
        t = t.to(self.device)
        context = context.to(self.device) if context is not None else None
        t_emb = self.time_embedding(t)
        x_emb = self.feature_embedding(x)
        context_emb = self.context_embedding(context) if context is not None else None
        h = torch.concat([x_emb, context_emb, t_emb], dim=1) if context is not None else torch.concat([x_emb, t_emb], dim=1) 
        return self.layers(h)

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
        self.dim_input = config.DIM_INPUT
        self.dim_output = config.DIM_INPUT
        self.dim_hidden = config.DIM_HIDDEN
        self.dim_context = config.DIM_CONTEXT
        self.dim_time_emb = config.DIM_TIME_EMB if config.TIME_ENCODING_TYPE is not None else 1
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT
        self.act_fn = get_activation_function(config.ACTIVATION)

        if config.TIME_ENCODING_TYPE == 'sinusoidal': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'randomfourier': self.time_embedding = nn.Sequential(GaussianFourierFeatures(self.dim_time_emb, scale=0.5),  nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'kolmogorov-arnold': self.time_embedding = nn.Sequential(KANLinear(1, self.dim_time_emb), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)                                                                
        elif config.TIME_ENCODING_TYPE is None: self.time_embedding = nn.Identity()

        self.kan = kan_block(dim_input=self.dim_input + self.dim_context + self.dim_time_emb, 
                            dim_output=self.dim_output, 
                            dim_hidden=self.dim_hidden, 
                            num_layers=self.num_layers, 
                            dropout=self.dropout, 
                            use_batch_norm=True)

    def forward(self, t, x, s=None, context=None):
        x = x.to(self.device)
        t = t.to(self.device)
        context = context.to(self.device) if context is not None else None
        t_emb = self.time_embedding(t)
        h = torch.concat([x, context, t_emb], dim=1) if context is not None else torch.concat([x, t_emb], dim=1) 
        return self.kan(h)



class MixedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.define_deep_models(config)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, config):

        self.dim_input_continuous = config.DIM_INPUT_CONTINUOUS
        self.dim_input_discrete = config.DIM_INPUT_DISCRETE
        self.vocab_size = config.VOCAB_SIZE

        self.dim_emb_time = config.DIM_EMB_TIME
        self.dim_emb_continuous = config.DIM_EMB_CONTINUOUS
        self.dim_emb_discrete = config.DIM_EMB_DISCRETE

        self.dim_hidden = config.DIM_HIDDEN
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT
        self.act_fn = get_activation_function(config.ACTIVATION)

        self.continous_embedding = nn.Linear(self.dim_input_continuous, self.dim_emb_continuous)
        self.discrete_embedding = nn.Embedding(self.vocab_size, self.dim_emb_discrete)


        self.backbone = fc_block(dim_input=self.dim_emb_continuous + self.dim_emb_discrete + self.dim_emb_time, 
                               dim_output=self.dim_hidden, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers // 2, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=True)
        
        self.continuous_head = fc_block(dim_input=self.dim_hidden, 
                               dim_output=self.dim_input_continuous, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers // 2, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=True)
        
        self.discrete_head = fc_block(dim_input=self.dim_hidden, 
                               dim_output=self.dim_input_discrete * self.vocab_size, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers // 2, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=True)
        

    def forward(self, t, x, s, context=None):

        x = x.to(self.device)
        s = s.to(self.device)
        t = t.to(self.device)

        # print('x={}, s={}, t={}'.format(x.shape, s.shape, t.shape))

        t_emb = sinusoidal_timestep_embedding(t, self.dim_emb_time, max_period=10000)
        x_emb = self.continous_embedding(x).unsqueeze(1)
        s_emb = self.discrete_embedding(s)
        
        print('x_emb={}, s_emb={}, t_emb={}'.format(x_emb.shape, s_emb.shape, t_emb.shape))

        h = torch.concat([x_emb, s_emb, t_emb], dim=-1) 

        h = h.squeeze(1)
        backbone = self.backbone(h)
        continuous_head = self.continuous_head(backbone)
        discrete_head = self.discrete_head(backbone)
        discrete_head = discrete_head.reshape(s.size(0), self.dim_input_discrete, self.vocab_size)

        return continuous_head, discrete_head

    def init_weights(self):
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
