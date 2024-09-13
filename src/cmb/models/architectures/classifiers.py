import torch
from torch import nn
from torch.nn.functional import softmax

from cmb.models.architectures.utils import (fc_block, 
                                            kan_block,
                                            get_activation_function, 
                                            KANLinear, KAN,
                                            SinusoidalPositionalEncoding,
                                            GaussianFourierFeatures)

class DiscreteStateMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.gamma = config.gamma
        self.time_epsilon = config.time_eps
        self.define_deep_models(config)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, config):
        self.dim_discrete = config.dim_discrete
        self.dim_discrete_emb = config.dim_discrete_emb
        self.dim_time_emb = config.dim_time_emb 
        self.dim_context = config.dim_context
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.act_fn = get_activation_function(config.activation)
        self.use_batch_norm = config.use_batch_norm

        #...Time embedding:

        if hasattr(config, 'time_embedding'):
            if config.time_embedding == 'sinusoidal': self.time_embedding = SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000)
            elif config.time_embedding == 'randomfourier': self.time_embedding = GaussianFourierFeatures(self.dim_time_emb, scale=0.5)
            elif config.time_embedding == 'kolmogorov-arnold': self.time_embedding = KANLinear(1, self.dim_time_emb)
            elif config.time_embedding == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)  
            else: raise NotImplementedError                                                              
        else:
            self.dim_time_emb = 1

        #...Discrete state embedding:
            
        self.state_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.dim_discrete_emb),
                                             nn.Flatten(start_dim=1), 
                                             KANLinear(self.dim_discrete * self.dim_discrete_emb, self.dim_discrete_emb)) 

        #...Context embedding:
            
        if hasattr(config, 'context_embedding') and self.dim_context > 0: 
            self.dim_context_emb = config.dim_context_emb
            if config.context_embedding == 'embedding': self.context_embedding = nn.Embedding(config.vocab_size, self.dim_context_emb)
            elif config.context_embedding == 'linear': self.time_embedding = nn.Linear(self.dim_context, self.dim_context_emb)  
            else: raise NotImplementedError                                                              
        else:
            self.dim_context_emb = self.dim_context if self.dim_context > 0 else 0
        
        #...MLP layers:
                        
        self.layers = fc_block(dim_input=self.dim_discrete_emb + self.dim_time_emb + self.dim_context_emb , 
                               dim_output=config.dim_discrete * config.vocab_size, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=self.use_batch_norm)


    def forward(self, t, k, context=None, mask=None, output_rates=False):
        
        t = t.to(self.device)
        k = k.to(self.device)
        t_emb = self.time_embedding(t) if hasattr(self, 'time_embedding') else t
        k_emb = self.state_embedding(k).squeeze(1)
        features = [t_emb, k_emb]
        
        if context is not None:
            context = context.to(self.device)
            context_emb = self.context_embedding(context) if hasattr(self, 'context_embedding') else context
            features.append(context_emb)

        if mask is not None:
            mask = mask.to(self.device)

        h = torch.concat(features, dim=1) 
        h = self.layers(h)
        logits = h.reshape(k.size(0), self.dim_discrete, self.vocab_size)

        if output_rates: 
            t = t.squeeze()
            t1 = 1. - self.time_epsilon
            beta_integral = (t1 - t) * self.gamma
            wt = torch.exp(-self.vocab_size * beta_integral)
            A = 1.0
            B = (wt * self.vocab_size) / (1. - wt)
            C = wt
            qx = softmax(logits, dim=2)
            qy = torch.gather(qx, 2, k.long().unsqueeze(2))
            rate = A + B[:, None, None] * qx + C[:, None, None] * qy
            return rate
        else:
            return logits

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)