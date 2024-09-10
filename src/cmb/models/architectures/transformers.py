
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmb.models.architectures.utils import (fc_block, 
                                            get_activation_function, 
                                            KANLinear, KAN,
                                            SinusoidalPositionalEncoding,
                                            GaussianFourierFeatures)

class ParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.particle_embedding = ParticleEmbedding(config)
        self.particle_attention_blocks = []
        self.particle_attention_blocks = ParticleAttentionBlock(dim_input=config.dim_hidden, 
                                                                         dim_output=config.dim_hidden, 
                                                                         dim_hidden=config.dim_hidden, 
                                                                         num_heads=config.num_heads, 
                                                                         activation=get_activation_function(config.activation), 
                                                                         dropout=config.dropout, 
                                                                         attention_embedding='linear') 
        self.projection = nn.Linear(config.dim_hidden, config.dim_continuous)
                                                
    def forward(self, t, x, k=None, context=None, mask=None):

        t = t.to(self.device) # time
        x = x.to(self.device) # continuous feature (b, n, dim_continuous) 
        k = k.to(self.device) if k is not None else None  # discrete feature (b, n, dim_discrete)          
        mask = torch.ones_like(x[...,0]).unsqueeze(-1) if mask is None else mask.unsqueeze(-1) 
        mask = mask.to(self.device)    
        h = self.particle_embedding(t=t, x=x, mask=mask)
        h_skip = h
        h = self.particle_attention_blocks(h, mask, skip_connection=h_skip)
        h = self.projection(h) * mask
        return h

class ParticleEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.time_embedding == 'sinusoidal': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(config.dim_time_emb, max_period=10000), nn.Linear(config.dim_time_emb, config.dim_time_emb))
        elif config.time_embedding == 'kolmogorov-arnold': self.time_embedding = nn.Sequential(KANLinear(1, config.dim_time_emb), nn.Linear(config.dim_time_emb, config.dim_time_emb))
        elif config.time_embedding is None: self.time_embedding = nn.Identity()
        else: raise NotImplementedError

        self.embedding = nn.Linear(config.dim_continuous + config.dim_time_emb, config.dim_hidden)

    def forward(self, t, x, k=None, mask=None):
        t_emb = self.time_embedding(t.squeeze(-1))
        t_emb = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        h = torch.concat([t_emb, x], dim=-1) 
        h = self.embedding(h)
        return h * mask

class ParticleAttentionBlock(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, num_heads=4, activation=nn.GELU(), dropout=0.0, attention_embedding='linear'):
        super().__init__()

        self.layernorm_0 = nn.LayerNorm(dim_input)
        self.mha_block = MultiHeadAttention(dim_input, dim_hidden, dim_hidden, num_heads, dropout, attention_embedding=attention_embedding)
        self.layernorm_1 = nn.LayerNorm(dim_hidden)
        self.layernorm_2 = nn.LayerNorm(dim_hidden)
        self.fc_block = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                      activation,
                                      nn.LayerNorm(dim_hidden),
                                      nn.Linear(dim_hidden, dim_output))
    def forward(self, x, mask, skip_connection=None):
        h = self.layernorm_0(x)
        h = self.mha_block(h, mask=mask)
        h = self.layernorm_1(h)
        h += x
        f = self.layernorm_2(h)
        f = self.fc_block(f) * mask
        f += h if skip_connection is None else skip_connection
        return f


# class GPT3Block(nn.Module):
#     """The GPT block."""

#     def __init__(self, config):
#         super().__init__()
#         self.device = config.device
#         if config.time_embedding == 'sinusoidal': self.time_embedding = nn.Sequential(SinusoidalPositionalEncoding(config.dim_time_emb, max_period=10000), nn.Linear(config.dim_time_emb, config.dim_time_emb))
#         elif config.time_embedding == 'kolmogorov-arnold': self.time_embedding = nn.Sequential(KANLinear(1, config.dim_time_emb), nn.Linear(config.dim_time_emb, config.dim_time_emb))
#         elif config.time_embedding is None: self.time_embedding = nn.Identity()
#         else: raise NotImplementedError

#         self.embedding = nn.Sequential(nn.Linear(config.dim_continuous + config.dim_discrete_emb + config.dim_time_emb, config.dim_hidden), 
#                                        nn.LayerNorm(config.dim_hidden))
        
#         self.att_block_1 = MultiHeadAttention(dim_input=config.dim_hidden, 
#                                             dim_output=config.dim_hidden,
#                                             dim_hidden=config.dim_hidden, 
#                                             num_heads=config.num_heads, 
#                                             dropout=config.dropout, 
#                                             attention_embedding=config.attention_embedding)
        
#         self.layernorm_1 = nn.LayerNorm(config.dim_hidden)

#         self.att_block_2 = MultiHeadAttention(dim_input=config.dim_hidden, 
#                                             dim_output=config.dim_hidden,
#                                             dim_hidden=config.dim_hidden, 
#                                             num_heads=config.num_heads, 
#                                             dropout=config.dropout, 
#                                             attention_embedding=config.attention_embedding)
#         self.layernorm_2 = nn.LayerNorm(config.dim_hidden)

#         self.att_block_3 = MultiHeadAttention(dim_input=config.dim_hidden, 
#                                             dim_output=config.dim_hidden,
#                                             dim_hidden=config.dim_hidden, 
#                                             num_heads=config.num_heads, 
#                                             dropout=config.dropout, 
#                                             attention_embedding=config.attention_embedding)                                            
#         self.layernorm_3 = nn.LayerNorm(config.dim_hidden)

#         self.layer_out = FeedForward(dim_input=config.dim_hidden, 
#                                 dim_hidden=config.dim_hidden,
#                                 dim_output=config.dim_continuous,
#                                 activation=get_activation_function(config.activation))
#         self.layernorm_out = nn.LayerNorm(config.dim_continuous)

#     def forward(self, t, x, k=None, context=None, mask=None):
#         # x: (b, n, dim)
        
#         t = t.to(self.device) # time
#         x = x.to(self.device) # continuous feature
#         mask = mask.to(self.device) if mask is not None else None

#         t_emb = self.time_embedding(t.squeeze(-1))
#         t_emb = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
#         h = torch.concat([t_emb, x], dim=-1) 

#         print(x.shape, h.shape, t_emb.shape)
#         h = self.embedding(h)

#         # if k is not None:
#         #     k = k.to(self.device) # discrete feature
#         #     k_emb = self.discrete_embedding(k)
#         #     h = torch.concat([h, k_emb], dim=1) 

#         h_res = h

#         h1 = self.att_block_1(h, mask=mask)
#         h1 += h_res
#         h1 = self.layernorm_1(h1)

#         h2 = self.att_block_2(h1, mask=mask)
#         h2 += h_res
#         h2 = self.layernorm_2(h2)

#         h3 = self.att_block_3(h2, mask=mask)
#         h3 += h_res
#         h3 = self.layernorm_3(h3)

#         h = self.layer_out(h3)
#         h = self.layernorm_out(h)

#         return h


# class FeedForward(nn.Module):
#     """Simple linear layer followed by a non-linearity to be placed after the attention blocks."""

#     def __init__(self, dim_input, dim_hidden, dim_output, activation=nn.ReLU()):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim_input, 4 * dim_hidden),
#             activation,
#             nn.Linear(4 * dim_hidden, dim_output),
#         )

#     def forward(self, x):
#         return self.net(x)

