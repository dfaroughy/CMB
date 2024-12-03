import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax
import numpy as np

from cmb.models.architectures.utils import (
    fc_blocks,
    get_activation_function,
    InputEmbeddings,
)

# ...Multi-Layer Perceptron architecture:


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.train.device

        # ...data dimensions:
        self.dim_features_continuous = config.data.dim.features.continuous
        self.dim_features_discrete = config.data.dim.features.discrete
        dim_context_continuous = config.data.dim.context.continuous
        self.vocab_size = config.data.vocab.size.features

        # ...embedding dimensions:
        dim_time_emb = config.model.dim.embed.time
        dim_features_continuous_emb = (
            config.model.dim.embed.features.continuous
            if config.model.dim.embed.features.continuous
            else self.dim_features_continuous
        )
        dim_features_discrete_emb = config.model.dim.embed.features.discrete
        dim_context_continuous_emb = (
            config.model.dim.embed.context.continuous
            if config.model.dim.embed.context.continuous
            else dim_context_continuous
        )
        dim_context_discrete_emb = config.model.dim.embed.context.discrete

        # ...components:
        self.embedding = InputEmbeddings(config)

        self.layers = fc_blocks(
            dim_input=dim_time_emb
            + dim_features_continuous_emb
            + dim_features_discrete_emb
            + dim_context_continuous_emb
            + dim_context_discrete_emb,
            dim_output=self.dim_features_continuous
            + self.dim_features_discrete * self.vocab_size,
            dim_hidden=config.model.dim.hidden,
            num_layers=config.model.num_layers,
            activation=get_activation_function(config.model.activation),
            dropout=config.model.dropout,
            use_batch_norm=config.model.use_batch_norm,
        )

        self.init_weights()

    def forward(
        self, t, x, k=None, context_continuous=None, context_discrete=None, mask=None
    ):
        t = t.to(self.device)
        x = x.to(self.device)
        k = k.to(self.device) if isinstance(k, torch.Tensor) else None
        context_continuous = (
            context_continuous.to(self.device)
            if isinstance(context_continuous, torch.Tensor)
            else None
        )
        context_discrete = (
            context_discrete.to(self.device)
            if isinstance(context_discrete, torch.Tensor)
            else None
        )
        mask = (
            mask.to(self.device)
            if isinstance(mask, torch.Tensor)
            else torch.ones_like(t).to(self.device)
        )
        h, _ = self.embedding(t, x, k, context_continuous, context_discrete, mask)
        h = self.layers(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


class ClassifierMLP(nn.Module):
    """MLP classifier for discrete models"""

    def __init__(self, config):
        super().__init__()
        self.dim_features_continuous = config.data.dim.features.continuous
        self.dim_features_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab.size.features

        self.mlp = MLP(config)

    def forward(
        self, t, k, x=None, context_continuous=None, context_discrete=None, mask=None
    ):
        h = self.mlp(t, x, k, context_continuous, context_discrete, mask)
        logits = h.reshape(k.size(0), self.dim_features_discrete, self.vocab_size)
        return logits


class HybridMLP(nn.Module):
    """MLP architecture for hybrid continuous-discrete models"""

    def __init__(self, config):
        super().__init__()
        self.dim_features_continuous = config.data.dim.features.continuous
        self.dim_features_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab.size.features

        self.mlp = MLP(config)

    def forward(
        self, t, x, k, context_continuous=None, context_discrete=None, mask=None
    ):
        h = self.mlp(t, x, k, context_continuous, context_discrete, mask)
        continuous_head = h[:, : self.dim_features_continuous]
        discrete_head = h[:, self.dim_features_continuous :]
        logits = discrete_head.reshape(
            k.size(0), self.dim_features_discrete, self.vocab_size
        )
        return continuous_head, logits
