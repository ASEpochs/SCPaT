import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(
            position * div_term[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.encoding[:, : x.size(1)]


class SemanticVectorEncoder(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.temporal_configurations = ((3, 1), (5, 2), (7, 3))
        self.input_projection = nn.Linear(1, dim)
        self.temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        groups=dim,
                    ),
                    nn.GELU(),
                    nn.Conv1d(dim, dim, kernel_size=1),
                )
                for kernel_size, dilation in self.temporal_configurations
            ]
        )
        self.fusion_score = nn.Linear(dim, 1)
        self.unit_score = nn.Linear(dim, 1)
        self.unit_projection = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.position_encoder = PositionalEncoder(dim) if pos else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = self.input_projection(x.unsqueeze(-1))
        base_channels = base.transpose(1, 2)
        branch_outputs = []
        for branch, (kernel_size, dilation) in zip(
            self.temporal_branches, self.temporal_configurations
        ):
            left_padding = (kernel_size - 1) * dilation
            branch_input = F.pad(base_channels, (left_padding, 0))
            branch_outputs.append(branch(branch_input).transpose(1, 2))
        multi_scale = torch.stack(branch_outputs, dim=1)
        fusion_logits = self.fusion_score(multi_scale).mean(dim=2)
        fusion_weights = torch.softmax(fusion_logits, dim=1).unsqueeze(2)
        fused = torch.sum(fusion_weights * multi_scale, dim=1)
        windows = fused.unfold(1, self.patch_len, self.stride)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        centers = windows.mean(dim=2, keepdim=True)
        variation = torch.mean(torch.abs(windows - centers), dim=-1, keepdim=True)
        attention = torch.softmax(self.unit_score(windows) + variation, dim=2)
        pooled = torch.sum(attention * windows, dim=2)
        units = pooled + self.unit_projection(pooled)
        if self.position_encoder is not None:
            units = units + self.position_encoder(units)
        return self.dropout(units)


class TransferEntropyGraphConstructor(nn.Module):
    def __init__(self, head_dim, units_per_variable, stability_penalty=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.units_per_variable = units_per_variable
        self.stability_penalty = stability_penalty
        self.source_projection = nn.Linear(head_dim, head_dim)
        self.history_projection = nn.Linear(head_dim, head_dim)
        self.future_projection = nn.Linear(head_dim, head_dim)
        self.transition_norm = nn.LayerNorm(head_dim)

    def _future_indices(self, length, device):
        indices = torch.arange(length, device=device)
        positions = indices % self.units_per_variable
        return torch.where(
            positions < self.units_per_variable - 1,
            indices + 1,
            indices,
        )

    def forward(self, h, alpha):
        batch_size, n_heads, length, _ = h.shape
        future_indices = self._future_indices(length, h.device)
        future = h.index_select(2, future_indices)
        source = torch.tanh(self.source_projection(h))
        transition = self.transition_norm(
            self.future_projection(future) + self.history_projection(h)
        )
        transition = torch.tanh(transition)
        scores = torch.einsum("bhid,bhjd->bhij", source, transition)
        scores = scores / math.sqrt(self.head_dim)
        dispersion = scores.std(dim=0, unbiased=False, keepdim=True)
        weights = F.softplus(scores - self.stability_penalty * dispersion)
        identity = torch.eye(length, device=h.device, dtype=h.dtype)
        weights = weights * (1.0 - identity.view(1, 1, length, length))
        keep_count = max(1, min(length - 1, math.ceil(float(alpha) * length)))
        top_indices = torch.topk(weights, keep_count, dim=-1).indices
        sparse_mask = torch.zeros_like(weights)
        sparse_mask.scatter_(-1, top_indices, 1.0)
        adjacency = weights * sparse_mask
        adjacency = adjacency + identity.view(1, 1, length, length)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class SemanticBlockAggregator(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.activation = nn.GELU()

    def forward(self, adjacency, h):
        batch_size, length, _ = h.shape
        head_dim = self.dim // self.n_heads
        values = self.value_projection(h)
        values = values.view(batch_size, length, self.n_heads, head_dim)
        values = values.permute(0, 2, 1, 3)
        aggregated = torch.matmul(adjacency, values)
        aggregated = aggregated.permute(0, 2, 1, 3).contiguous()
        aggregated = aggregated.view(batch_size, length, self.dim)
        return self.activation(self.output_projection(aggregated))


class TransformerExpert(nn.Module):
    def __init__(self, dim, d_ff, n_heads, dropout):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        return self.encoder(x)


class ImportanceAwareRouter(nn.Module):
    def __init__(
        self,
        dim,
        d_ff,
        n_heads,
        top_p=0.5,
        num_experts=3,
        dropout=0.0,
    ):
        super().__init__()
        self.top_p = min(max(float(top_p), 1e-6), 1.0)
        self.num_experts = num_experts
        self.routing_projection = nn.Linear(dim * 2 + 2, dim)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.noise = nn.Linear(dim, num_experts, bias=False)
        self.semantic_bias = nn.Linear(dim, dim)
        self.base_bias = nn.Parameter(torch.zeros(dim))
        self.experts = nn.ModuleList(
            [
                TransformerExpert(dim, d_ff, n_heads, dropout)
                for _ in range(num_experts)
            ]
        )

    def _top_p_weights(self, probabilities):
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, dim=-1, descending=True
        )
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        keep_sorted = cumulative - sorted_probabilities < self.top_p
        sparse_sorted = sorted_probabilities * keep_sorted
        sparse = torch.zeros_like(probabilities)
        sparse.scatter_(-1, sorted_indices, sparse_sorted)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _routing_loss(self, probabilities, weights):
        importance = probabilities.mean(dim=(0, 1))
        load = (weights > 0).to(probabilities.dtype).mean(dim=(0, 1))
        importance_loss = importance.var(unbiased=False) / (
            importance.mean().square() + 1e-8
        )
        load_loss = load.var(unbiased=False) / (load.mean().square() + 1e-8)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=-1
        ).mean()
        return importance_loss + load_loss + 0.01 * entropy

    def forward(self, h, block_context, adjacency, is_training=False):
        mean_adjacency = adjacency.mean(dim=1)
        incoming_strength = mean_adjacency.sum(dim=-2)
        neighborhood_density = (mean_adjacency > 0).to(h.dtype).mean(dim=-1)
        statistics = torch.stack(
            [incoming_strength, neighborhood_density], dim=-1
        )
        routing_input = torch.cat([h, block_context, statistics], dim=-1)
        routing_representation = torch.tanh(
            self.routing_projection(routing_input)
        )
        logits = self.gate(routing_representation)
        if is_training:
            noise_scale = F.softplus(self.noise(routing_representation)) + 1e-2
            logits = logits + torch.randn_like(logits) * noise_scale
        probabilities = torch.softmax(logits, dim=-1)
        weights = self._top_p_weights(probabilities)
        bias = self.base_bias + torch.tanh(
            self.semantic_bias(routing_representation)
        )
        expert_input = h + block_context + bias
        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=2
        )
        routed = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=2)
        return routed, self._routing_loss(probabilities, weights)


class SCPaTLayer(nn.Module):
    def __init__(
        self,
        dim,
        units_per_variable,
        d_ff=None,
        n_heads=4,
        top_p=0.5,
        dropout=0.0,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        d_ff = dim * 4 if d_ff is None else d_ff
        self.dim = dim
        self.n_heads = n_heads
        self.graph_constructor = TransferEntropyGraphConstructor(
            dim // n_heads, units_per_variable
        )
        self.block_aggregator = SemanticBlockAggregator(dim, n_heads)
        self.router = ImportanceAwareRouter(
            dim, d_ff, n_heads, top_p=top_p, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim),
        )

    def forward(self, z, alpha=0.1, is_training=False):
        batch_size, length, _ = z.shape
        normalized = self.norm1(z)
        multi_head = normalized.view(
            batch_size, length, self.n_heads, self.dim // self.n_heads
        )
        multi_head = multi_head.permute(0, 2, 1, 3)
        adjacency = self.graph_constructor(multi_head, alpha)
        block_context = self.block_aggregator(adjacency, normalized)
        routed, router_loss = self.router(
            normalized, block_context, adjacency, is_training
        )
        z = z + self.dropout(routed)
        z = z + self.dropout(self.feed_forward(self.norm2(z)))
        return z, router_loss


class SCPaTEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        units_per_variable,
        d_ff=None,
        n_heads=4,
        n_blocks=3,
        top_p=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SCPaTLayer(
                    hidden_dim,
                    units_per_variable,
                    d_ff,
                    n_heads,
                    top_p,
                    dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, z, alpha=0.1, is_training=False):
        losses = []
        for layer in self.layers:
            z, router_loss = layer(z, alpha, is_training)
            losses.append(router_loss)
        return z, torch.stack(losses).mean()


class InstanceNormalizer(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"Unsupported normalization mode: {mode}")

    def _get_statistics(self, x):
        reduction_dims = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = x.mean(dim=reduction_dims, keepdim=True).detach()
        variance = x.var(dim=reduction_dims, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(variance + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps**2)
        x = x * self.stdev
        return x + (self.last if self.subtract_last else self.mean)


class GlobalPredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, z_encoded):
        return self.projection(z_encoded)
