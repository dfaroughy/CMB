import torch
from torch import nn

from cmb.models.architectures.utils import (fc_block, 
                                            kan_block,
                                            get_activation_function, 
                                            PermutationLayer,
                                            KANLinear, KAN,
                                            SinusoidalPositionalEncoding,
                                            GaussianFourierFeatures)
class StateClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.define_deep_models(config)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, config):
        self.dim_input = config.DIM_INPUT
        self.vocab_size = config.VOCAB_SIZE
        self.dim_output = config.DIM_INPUT * config.VOCAB_SIZE
        self.dim_hidden = config.DIM_HIDDEN
        self.dim_context = config.DIM_CONTEXT        
        self.dim_state_emb = config.DIM_STATE_EMB if config.STATE_ENCODING_TYPE is not None else config.DIM_INPUT
        self.dim_time_emb = config.DIM_TIME_EMB if config.TIME_ENCODING_TYPE is not None else 1
        self.dim_context_emb = config.DIM_CONTEXT_EMB if config.CONTEXT_ENCODING_TYPE is not None else config.DIM_CONTEXT
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT
        self.act_fn = get_activation_function(config.ACTIVATION)

        if config.STATE_ENCODING_TYPE == 'linear': self.state_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.dim_state_emb),
                                                                                        nn.Flatten(start_dim=1), 
                                                                                        nn.Linear(self.dim_input * self.dim_state_emb, self.dim_state_emb))   
        elif config.STATE_ENCODING_TYPE == 'kolmogorov-arnold': self.state_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.dim_state_emb), 
                                                                                                     nn.Flatten(start_dim=1), 
                                                                                                     KANLinear(self.dim_input * self.dim_state_emb, self.dim_state_emb))                     
        elif config.STATE_ENCODING_TYPE is None: self.state_embedding = nn.Identity()
        else: raise NotImplementedError 

        if config.TIME_ENCODING_TYPE == 'sinusoidal': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'randomfourier': self.time_embedding = nn.Sequential(GaussianFourierFeatures(self.dim_time_emb, scale=4),  nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'kolmogorov-arnold': self.time_embedding = nn.Sequential(KANLinear(1, self.dim_time_emb), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.TIME_ENCODING_TYPE == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)                                                                
        elif config.TIME_ENCODING_TYPE is None: self.time_embedding = nn.Identity()
        else: raise NotImplementedError

        if config.CONTEXT_ENCODING_TYPE == 'linear': self.context_embedding = nn.Linear(self.dim_context, self.dim_context_emb)   
        elif config.CONTEXT_ENCODING_TYPE == 'kolmogorov-arnold': self.context_embedding = nn.Sequential(KANLinear(self.dim_context, self.dim_context_emb), nn.Linear(self.dim_context_emb, self.dim_context_emb))                     
        elif config.CONTEXT_ENCODING_TYPE is None: self.context_embedding = nn.Identity()
        else: raise NotImplementedError 

        self.layers = fc_block(dim_input=self.dim_state_emb + self.dim_context_emb + self.dim_time_emb, 
                            dim_output=self.dim_output, 
                            dim_hidden=self.dim_hidden, 
                            num_layers=self.num_layers, 
                            activation=self.act_fn, 
                            dropout=self.dropout, 
                            use_batch_norm=True)

    def forward(self, t, k, x=None, context=None):
        k = k.to(self.device)
        t = t.to(self.device)
        context = context.to(self.device) if context is not None else None
        
        time_emb = self.time_embedding(t)
        state_emb = self.state_embedding(k)
        context_emb = self.context_embedding(context) if context is not None else None

        h = torch.concat([state_emb, context_emb, time_emb], dim=1) if context is not None else torch.concat([state_emb, time_emb], dim=1) 
        h = self.layers(h)
        rate_logits = h.reshape(k.size(0), self.dim_input, self.vocab_size)

        return rate_logits

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

