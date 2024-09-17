import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax
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
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.define_deep_models(config)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, config):
        self.dim_continuous = config.dim_continuous
        self.dim_context = config.dim_context
        self.dim_continuous_emb = config.dim_continuous_emb
        self.dim_time_emb = config.dim_time_emb 
        self.dim_context_emb = config.dim_context_emb
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

        #...Continuous embedding:
        if hasattr(config, 'continuous_embedding'):
            if config.continuous_embedding == 'linear': self.continuous_embedding = nn.Linear(self.dim_continuous, self.dim_continuous_emb)
            elif config.continuous_embedding == 'kolmogorov-arnold': self.continuous_embedding = KANLinear(self.dim_continuous, self.dim_continuous_emb)
            else: raise NotImplementedError
        else:
            self.dim_continuous_emb = self.dim_continuous

        #...Context embedding:
            
        if hasattr(config, 'context_embedding') and self.dim_context > 0: 
            self.dim_context_emb = config.dim_context_emb
            if config.context_embedding == 'embedding': self.context_embedding = nn.Embedding(config.vocab_size, self.dim_context_emb)
            elif config.context_embedding == 'linear': self.context_embedding = nn.Linear(self.dim_context, self.dim_context_emb)  
            else: raise NotImplementedError                                                              
        else:
            self.dim_context_emb = self.dim_context if self.dim_context > 0 else 0
        
        #...MLP layers:
                        
        self.layers = fc_block(dim_input=self.dim_continuous_emb + self.dim_time_emb + self.dim_context_emb , 
                               dim_output=self.dim_continuous, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=self.use_batch_norm)


    def forward(self, t, x, context=None, mask=None):
        
        t = t.to(self.device)
        x = x.to(self.device)
        t_emb = self.time_embedding(t) if hasattr(self, 'time_embedding') else t
        x_emb = self.continuous_embedding(x)
        features = [t_emb, x_emb]
        
        if context is not None:
            context = context.to(self.device)
            context_emb = self.context_embedding(context) if hasattr(self, 'context_embedding') else context
            features.append(context_emb)

        if mask is not None:
            mask = mask.to(self.device)

        h = torch.concat(features, dim=1) 
        h = self.layers(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)



class MixedDataMLP(nn.Module):
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
        self.dim_continuous = config.dim_continuous
        self.dim_discrete = config.dim_discrete
        self.dim_context = config.dim_context
        
        self.dim_continuous_emb = config.dim_continuous_emb
        self.dim_discrete_emb = config.dim_discrete_emb
        self.dim_time_emb = config.dim_time_emb 
        self.dim_context_emb = config.dim_context_emb
        
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

        #...Continuous embedding:
        if hasattr(config, 'continuous_embedding'):
            if config.continuous_embedding == 'linear': self.continuous_embedding = nn.Linear(self.dim_continuous, self.dim_continuous_emb)
            elif config.continuous_embedding == 'kolmogorov-arnold': self.continuous_embedding = KANLinear(self.dim_continuous, self.dim_continuous_emb)
            else: raise NotImplementedError
        else:
            self.dim_continuous_emb = self.dim_continuous

        #...Discrete state embedding:
            
                #...Continuous embedding:
        if hasattr(config, 'discrete_embedding'):
            if config.discrete_embedding == 'linear': self.discrete_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.dim_discrete_emb),
                                                                                        nn.Flatten(start_dim=1), 
                                                                                        nn.Linear(self.dim_discrete * self.dim_discrete_emb, self.dim_discrete_emb)) 
                
            elif config.discrete_embedding == 'kolmogorov-arnold': self.discrete_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.dim_discrete_emb),
                                                                                       nn.Flatten(start_dim=1), 
                                                                                       KANLinear(self.dim_discrete * self.dim_discrete_emb, self.dim_discrete_emb)) 
            else: raise NotImplementedError
        else:
            self.dim_discrete_emb = self.dim_discrete


        #...Context embedding:
            
        if hasattr(config, 'context_embedding') and self.dim_context > 0: 
            self.dim_context_emb = config.dim_context_emb
            if config.context_embedding == 'embedding': self.context_embedding = nn.Embedding(config.vocab_size, self.dim_context_emb)
            elif config.context_embedding == 'linear': self.context_embedding = nn.Linear(self.dim_context, self.dim_context_emb)  
            else: raise NotImplementedError                                                              
        else:
            self.dim_context_emb = self.dim_context if self.dim_context > 0 else 0
        
        #...MLP layers:
                        
        self.layers = fc_block(dim_input=self.dim_continuous_emb + self.dim_discrete_emb + self.dim_time_emb + self.dim_context_emb , 
                               dim_output=self.dim_continuous + config.dim_discrete * config.vocab_size, 
                               dim_hidden=self.dim_hidden, 
                               num_layers=self.num_layers, 
                               activation=self.act_fn, 
                               dropout=self.dropout, 
                               use_batch_norm=self.use_batch_norm)


    def forward(self, t, x, k, context=None, mask=None, output_rates=False):
        
        t = t.to(self.device)
        x = x.to(self.device)
        k = k.to(self.device)
        t_emb = self.time_embedding(t) if hasattr(self, 'time_embedding') else t
        x_emb = self.continuous_embedding(x)
        k_emb = self.discrete_embedding(k).squeeze(1)

        features = [t_emb, x_emb, k_emb]
        
        if context is not None:
            context = context.to(self.device)
            context_emb = self.context_embedding(context) if hasattr(self, 'context_embedding') else context
            features.append(context_emb)

        if mask is not None:
            mask = mask.to(self.device)

        h = torch.concat(features, dim=1) 
        h = self.layers(h)
        continuous_head = h[:, :self.dim_continuous]
        discrete_head = h[:, self.dim_continuous:]
        logits = discrete_head.reshape(k.size(0), self.dim_discrete, self.vocab_size)

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
            return continuous_head, rate
        else:
            return continuous_head, logits

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)




# class MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.device = config.device
#         self.vocab_size = config.vocab_size
#         self.define_deep_models(config)
#         self.init_weights()
#         self.to(self.device)

#     def define_deep_models(self, config):
#         self.dim_continuous = config.dim_continuous
#         self.dim_discrete = config.dim_discrete
#         self.dim_context = config.dim_context
#         self.dim_hidden = config.dim_hidden
#         self.dim_time_emb = config.dim_time_emb 
#         self.num_layers = config.num_layers
#         self.dropout = config.dropout
#         self.act_fn = get_activation_function(config.activation)
#         self.use_batch_norm = config.use_batch_norm

#         #...Time embedding:

#         if hasattr(config, 'time_embedding'):
#             if config.time_embedding == 'sinusoidal': self.time_embedding = SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000)
#             elif config.time_embedding == 'randomfourier': self.time_embedding = GaussianFourierFeatures(self.dim_time_emb, scale=0.5)
#             elif config.time_embedding == 'kolmogorov-arnold': self.time_embedding = KANLinear(1, self.dim_time_emb)
#             elif config.time_embedding == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)  
#             else: raise NotImplementedError                                                              
#         else:
#             self.dim_time_emb = 1

#         #...Discrete embedding:
            
#         if self.dim_discrete > 0: 
#             self.dim_discrete_emb = config.dim_discrete_emb
#             self.discrete_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.dim_discrete_emb),
#                                                     nn.Flatten(start_dim=1), 
#                                                     nn.Linear(self.dim_discrete * self.dim_discrete_emb, self.dim_discrete_emb)) 
#         else:
#             self.dim_discrete_emb = 0

#         #...Context embedding:
            
#         if hasattr(config, 'context_embedding') and self.dim_context > 0: 
#             self.dim_context_emb = config.dim_context_emb
#             if config.context_embedding == 'embedding': self.context_embedding = nn.Embedding(config.vocab_size, self.dim_context_emb)
#             elif config.context_embedding == 'linear': self.time_embedding = nn.Linear(self.dim_context, self.dim_context_emb)  
#             else: raise NotImplementedError                                                              
#         else:
#             self.dim_context_emb = self.dim_context if self.dim_context > 0 else 0
        
#         #...MLP layers:
                        
#         self.layers = fc_block(dim_input=self.dim_continuous + self.dim_context_emb + self.dim_discrete_emb + self.dim_time_emb, 
#                                dim_output=self.dim_continuous + config.dim_discrete * config.vocab_size, 
#                                dim_hidden=self.dim_hidden, 
#                                num_layers=self.num_layers, 
#                                activation=self.act_fn, 
#                                dropout=self.dropout, 
#                                use_batch_norm=self.use_batch_norm)


#     def forward(self, t, x=None, k=None, context=None, mask=None):
        
#         t = t.to(self.device)
#         t = self.time_embedding(t) if hasattr(self, 'time_embedding') else t
#         features = [t]
        
#         if x is not None:
#             x = x.to(self.device)
#             features.append(x)

#         if k is not None:
#             k = k.to(self.device)
#             k = self.discrete_embedding(k).squeeze(1)
#             features.append(k)

#         if context is not None:
#             context = context.to(self.device)
#             context = self.context_embedding(context) if hasattr(self, 'context_embedding') else context
#             features.append(context)

#         if mask is not None:
#             mask = mask.to(self.device)

#         h = torch.concat(features, dim=1) 
#         h = self.layers(h)

#         if self.dim_continuous * self.dim_discrete==0:
#             return h if self.dim_continuous > 0 else h.reshape(k.size(0), self.dim_discrete, self.vocab_size)
#         else:
#             continuous = h[:, :self.dim_continuous]
#             discrete = h[:, self.dim_continuous:]
#             discrete = discrete.reshape(k.size(0), self.dim_discrete, self.vocab_size)
#             return continuous, discrete

#     def init_weights(self):
#         for layer in self.layers:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
    

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
            if config.time_embedding == 'sinusoidal': self.time_embedding = SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000)
            elif config.time_embedding == 'randomfourier': self.time_embedding = GaussianFourierFeatures(self.dim_time_emb, scale=0.5)
            elif config.time_embedding == 'kolmogorov-arnold': self.time_embedding = KANLinear(1, self.dim_time_emb)
            elif config.time_embedding == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)  
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
            if config.context_embedding == 'embedding': self.context_embedding = nn.Embedding(config.vocab_size, self.dim_context_emb)
            elif config.context_embedding == 'linear': self.time_embedding = nn.Linear(self.dim_context, self.dim_context_emb)  
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

        if config.time_embedding == 'sinusoidal': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(self.dim_time_emb, max_period=10000), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.time_embedding == 'randomfourier': self.time_embedding = nn.Sequential(GaussianFourierFeatures(self.dim_time_emb, scale=0.5),  nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.time_embedding == 'kolmogorov-arnold': self.time_embedding = nn.Sequential(KANLinear(1, self.dim_time_emb), nn.Linear(self.dim_time_emb, self.dim_time_emb))
        elif config.time_embedding == 'linear': self.time_embedding = nn.Linear(1, self.dim_time_emb)                                                                
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
    
