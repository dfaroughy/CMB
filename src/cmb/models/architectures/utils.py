import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_activation_function(name: str = "ReLU"):
    if name is not None:
        activation_functions = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU(),
            "SELU": nn.SELU(),
            "GLU": nn.GLU(),
            "GELU": nn.GELU(),
            "CELU": nn.CELU(),
            "PReLU": nn.PReLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "Hardswish": nn.Hardswish(),
            "Hardtanh": nn.Hardtanh(),
            "LogSigmoid": nn.LogSigmoid(),
            "Softplus": nn.Softplus(),
            "Softsign": nn.Softsign(),
            "Softshrink": nn.Softshrink(),
            "Softmin": nn.Softmin(),
            "Softmax": nn.Softmax(),
        }
        return activation_functions[name]
    else:
        return None


def fc_blocks(
    dim_input,
    dim_output,
    dim_hidden,
    num_layers,
    activation,
    dropout,
    use_batch_norm=False,
):
    BatchNorm = nn.BatchNorm1d if use_batch_norm else nn.Identity

    layers = [nn.Linear(dim_input, dim_hidden), BatchNorm(dim_hidden), activation]
    if dropout:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        layers.extend(
            [nn.Linear(dim_hidden, dim_hidden), BatchNorm(dim_hidden), activation]
        )
        if dropout:
            layers.extend([nn.Dropout(dropout)])

    layers.append(nn.Linear(dim_hidden, dim_output))
    return nn.Sequential(*layers)


def kan_blocks(
    dim_input, dim_output, dim_hidden, num_layers, dropout, use_batch_norm=False
):
    BatchNorm = nn.BatchNorm1d if use_batch_norm else nn.Identity

    layers = [KANLinear(dim_input, dim_hidden), BatchNorm(dim_hidden)]
    if dropout:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        layers.extend([KANLinear(dim_hidden, dim_hidden), BatchNorm(dim_hidden)])
        if dropout:
            layers.extend([nn.Dropout(dropout)])

    layers.append(nn.Linear(dim_hidden, dim_output))
    return nn.Sequential(*layers)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        dim_hidden=128,
        num_heads=4,
        dropout=0.0,
        attention_embedding="linear",
    ):
        super().__init__()

        assert (
            dim_hidden % num_heads == 0
        ), "hidden dimension must be divisible by number of heads"
        self.dim_head = dim_hidden // num_heads
        self.num_head = num_heads
        self.dim_hidden = dim_hidden
        self.register_buffer("tril", torch.tril(torch.ones(dim_hidden, dim_hidden)))

        if attention_embedding == "linear":
            self.k = nn.Linear(dim_input, dim_hidden, bias=False)
            self.q = nn.Linear(dim_input, dim_hidden, bias=False)
            self.v = nn.Linear(dim_input, dim_hidden, bias=False)
        elif attention_embedding == "kolmogorov-arnold":
            self.k = KANLinear(dim_input, dim_hidden)
            self.q = KANLinear(dim_input, dim_hidden)
            self.v = KANLinear(dim_input, dim_hidden)

        self.proj = nn.Linear(dim_hidden, dim_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, dim = x.shape
        K, V, Q = self.k(x), self.v(x), self.q(x)  # (B, T, E)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, n, d) -> (b, n, num_heads, head_dim)
        K = K.view(b, n, self.num_head, self.dim_head)
        V = V.view(b, n, self.num_head, self.dim_head)
        Q = Q.view(b, n, self.num_head, self.dim_head)

        # Transpose: (b, n, num_head, head_dim) -> (b, num_head, n, head_dim)
        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute scaled dot-product attention
        # (b, num_head, n, head_dim) @ (b, num_head, head_dim, n) -> (b, num_head,n, n)
        QK = Q @ K.transpose(2, 3) * K.shape[-1] ** -0.5

        if mask is not None:
            mask = mask.expand(-1, -1, n)  # (b, n) -> (b, n, n)
            # (b, n, n) -> (b, num_head, n, n)
            mask = mask.unsqueeze(1).expand(b, self.num_head, n, n)
            # Need to set a finite number for the masking, instead of -inf,
            # otherwise softmax results in nans.
            # (b, num_head, n, n)
            QK = QK.masked_fill(mask == 0, float("-1e9"))

        # Apply the causal mask, cropped to the sequence length
        # (b, num_head, n, n)
        QK = QK.masked_fill(self.tril[:n, :n] == 0, float("-inf"))

        A = F.softmax(QK, dim=-1)  # (B, num_head, T, T)
        A = self.dropout(A)

        # attn_weights have shape (b, num_head, n, n) and V (b, num_head, n, head_dim)
        # (b, num_head, n, head_dim) -> (b, n, num_head, head_dim)
        context_vec = (A @ V).transpose(1, 2)

        # Combine heads, where dim_hidden = num_head * dim_head
        context_vec = context_vec.contiguous().view(b, n, self.dim_hidden)
        context_vec = self.proj(context_vec)

        return context_vec


class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super(InputEmbeddings, self).__init__()

        # ...dimensions:
        dim_features_continuous = config.data.dim.features.continuous
        dim_features_discrete = config.data.dim.features.discrete
        dim_context_continuous = config.data.dim.context.continuous
        dim_context_discrete = config.data.dim.context.discrete

        # ...vocab sizes for discrete data:
        vocab_size = config.data.vocab.size.features
        vocab_size_context = config.data.vocab.size.context

        # ...embedding types:
        embed_type_time = config.model.embed_type.time
        embed_type_features_continuous = config.model.embed_type.features.continuous
        embed_type_features_discrete = config.model.embed_type.features.discrete
        embed_type_context_continuous = config.model.embed_type.context.continuous
        embed_type_context_discrete = config.model.embed_type.context.discrete

        # ...embedding dimensions:
        dim_time_emb = config.model.dim.embed.time
        dim_features_continuous_emb = (
            config.model.dim.embed.features.continuous
            if config.model.dim.embed.features.continuous
            else dim_features_continuous
        )
        dim_features_discrete_emb = config.model.dim.embed.features.discrete
        dim_context_continuous_emb = (
            config.model.dim.embed.context.continuous
            if config.model.dim.embed.context.continuous
            else dim_context_continuous
        )
        dim_context_discrete_emb = config.model.dim.embed.context.discrete

        # ...Time embeddings:

        if embed_type_time == "SinusoidalPositionalEncoding":
            self.time_embedding = SinusoidalPositionalEncoding(
                dim_time_emb, max_period=10000
            )
        elif embed_type_time == "KANLinear":
            self.time_embedding = KANLinear(1, dim_time_emb)
        elif embed_type_time == "Linear":
            self.time_embedding = nn.Linear(1, dim_time_emb)
        else:
            NotImplementedError(
                "Time embedding not implemented, choose from `SinusoidalPositionalEncoding`, `KANLinear` or `Linear`"
            )

        # ...Feature embeddings:

        if dim_features_continuous_emb:
            if embed_type_features_continuous == "Linear":
                self.embedding_continuous = nn.Linear(
                    dim_features_continuous, dim_features_continuous_emb
                )
            elif embed_type_features_continuous is None:
                self.embedding_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous features embedding not implemented, choose from `kolmogorov-arnold`, `linear` or None"
                )

        if dim_features_discrete:
            if embed_type_features_discrete == "Embedding":
                self.embedding_discrete = nn.Embedding(
                    vocab_size, dim_features_discrete_emb
                )
            elif embed_type_features_discrete == "Linear":
                self.embedding_discrete = nn.Linear(
                    dim_features_discrete, dim_features_continuous_emb
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

        # ...Context embeddings:

        if dim_context_continuous:
            if embed_type_context_continuous == "Embedding":
                self.embedding_context_continuous = nn.Linear(
                    dim_context_continuous, dim_context_continuous_emb
                )
            elif embed_type_context_continuous is None:
                self.embedding_context_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous context embedding not implemented, use `embedding` or None"
                )

        if dim_context_discrete:
            if embed_type_context_discrete == "Embedding":
                self.embedding_context_discrete = nn.Embedding(
                    vocab_size_context, dim_context_discrete_emb
                )
            elif embed_type_context_discrete == "Linear":
                self.embedding_context_discrete = nn.Linear(
                    dim_context_discrete, dim_features_continuous_emb
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

    def forward(
        self, t, x, k, context_continuous=None, context_discrete=None, mask=None
    ):
        """
        Forward pass of the particle embedding.

        Arguments:
        - t: Time input of shape (batch_size, 1) or (batch_size, 1, 1)
        - x: Particle continuous features of shape (batch_size, max_num_particles, dim_continuous)
        - k: Particle discrete features of shape (batch_size, max_num_particles, dim_discrete)
        - context_continuous: Continuous context features of shape (batch_size, dim_context_continuous)
        - context_discrete: Discrete context features of shape (batch_size, dim_context_discrete)
        - mask: Binary mask of shape (batch_size, max_num_particles, 1) indicating valid particles (1) or masked particles (0)

        Returns:
        - h: Embedded particles of shape (batch_size, N, dim_hidden), masked appropriately
        - context: Embedded context of shape (batch_size, dim_context)
        """

        # ...time:

        t_emb = self.time_embedding(t.squeeze(-1))
        t_context_emb = t_emb.clone().to(t_emb.device)
        if x.ndim == 3:
            t_emb = t_emb.unsqueeze(1).repeat(
                1, x.shape[1], 1
            )  # (b, dim_time_emb) -> (b, n, dim_time_emb)

        features = [t_emb]
        context = [t_context_emb]

        # ...features:

        if hasattr(self, "embedding_continuous"):
            emb = self.embedding_continuous(x)
            features.append(emb)

        if hasattr(self, "embedding_discrete"):
            emb = self.embedding_discrete(k.squeeze(-1))
            if x.ndim == 2:
                emb = emb.squeeze(1)
            features.append(emb)

        # ...context:

        if hasattr(self, "embedding_context_continuous"):
            emb = self.embedding_context_continuous(context_continuous)
            context.append(emb)

        if hasattr(self, "embedding_context_discrete"):
            emb = self.embedding_context_discrete(context_discrete).squeeze(1)
            context.append(emb)

        features = torch.cat(
            features, dim=-1
        )  # (b, n, dim_continuous_emb + dim_discrete_emb + dim_time_emb)
        context = torch.cat(
            context, dim=-1
        )  # (b, dim_context_continuous_emb + dim_context_discrete_emb + dim_time_emb)

        return features * mask, context


class PermutationLayer(nn.Module):
    def __init__(self, *dims):
        super(PermutationLayer, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding with log-linear spaced frequencies for each dimension"""

    def __init__(self, dim, max_period=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding.squeeze()


class GaussianFourierFeatures(nn.Module):
    """Random Gaussian features for encoding time steps.
    Inspired by https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, dim, scale=10.0):
        super().__init__()
        half = dim // 2
        self.w = nn.Parameter(torch.randn(half) * scale, requires_grad=False)
        self.dim = dim

    def forward(self, t):
        self.w = self.w.to(t.device)
        t_proj = 2 * math.pi * t[..., None] * self.w[None, ...]
        embedding = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding.squeeze()


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (self.grid).to(
            x.device
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        self.base_weight = self.base_weight.to(x.device)
        self.spline_weight = self.spline_weight.to(x.device)
        self.spline_scaler = self.spline_scaler.to(x.device)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
