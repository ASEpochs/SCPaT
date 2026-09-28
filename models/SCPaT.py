import torch.nn as nn

from layers.SCPaT_layers import (
    GlobalPredictionHead,
    InstanceNormalizer,
    SCPaTEncoder,
    SemanticVectorEncoder,
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = (
            configs.stride
            if hasattr(configs, "stride") and configs.stride is not None
            else self.patch_len
        )
        self.num_units = (self.seq_len - self.patch_len) // self.stride + 1
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p
        if configs.enc_in != self.n_vars:
            raise ValueError("enc_in must equal c_out")
        if self.patch_len > self.seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        self.normalizer = InstanceNormalizer(
            configs.enc_in, affine=True, subtract_last=False
        )
        self.semantic_vector_encoder = SemanticVectorEncoder(
            self.dim,
            self.patch_len,
            self.stride,
            bool(configs.pos),
            configs.dropout,
        )
        self.scpat_encoder = SCPaTEncoder(
            self.dim,
            self.num_units,
            configs.d_ff,
            configs.n_heads,
            configs.e_layers,
            self.top_p,
            configs.dropout,
        )
        self.prediction_head = GlobalPredictionHead(
            self.dim * self.num_units, self.pred_len
        )

    def forward(self, z_in, masks=None, is_training=False, target=None):
        batch_size, input_length, n_vars = z_in.shape
        if input_length != self.seq_len or n_vars != self.n_vars:
            raise ValueError(
                f"Expected input shape [B, {self.seq_len}, {self.n_vars}], "
                f"received {tuple(z_in.shape)}"
            )
        normalized = self.normalizer(z_in, "norm")
        channel_sequences = normalized.permute(0, 2, 1).reshape(
            batch_size * n_vars, input_length
        )
        semantic_units = self.semantic_vector_encoder(channel_sequences)
        semantic_units = semantic_units.view(
            batch_size, n_vars * self.num_units, self.dim
        )
        encoded, router_loss = self.scpat_encoder(
            semantic_units, self.alpha, is_training
        )
        encoded = encoded.view(
            batch_size, n_vars, self.num_units, self.dim
        ).flatten(start_dim=2)
        forecast = self.prediction_head(encoded).permute(0, 2, 1)
        forecast = self.normalizer(forecast, "denorm")
        return forecast, router_loss
