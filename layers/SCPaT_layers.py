import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.normal import Normal

def static_sparsification(x, alpha=0.5, largest=False):
    k = int(alpha * x.shape[-1])
    if k == 0:
        return torch.ones_like(x, dtype=torch.float32)
        
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)
    return mask


class SemanticBlockAggregator(nn.Module):

    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.activation = nn.GELU()

    def forward(self, A, h):
        B, L, D = h.shape
        H = self.n_heads
        D_head = D // H
        h_proj = self.proj(h).view(B, L, H, D_head).permute(0, 2, 1, 3)
        A_flat = A.reshape(B * H, L, L)
        h_flat = h_proj.reshape(B * H, L, D_head)
        out_flat = torch.bmm(A_flat, h_flat) 
        out = out_flat.view(B, H, L, D_head).permute(0, 2, 1, 3).contiguous().view(B, L, D) # [B, L, D]
        return self.activation(out)


class ImportanceAwareRouter(nn.Module):

    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p
        
        self.calculate_cv_squared = self._cv_squared
        self.calculate_cross_entropy = self._cross_entropy

    def _cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = ((self.softplus(raw_noise) + noise_epsilon))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        routing_entropy_loss = self.calculate_cross_entropy(logits)
        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p
        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask
        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1
        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        load_balancing_loss = self.calculate_cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = load_balancing_loss + lambda_2 * routing_entropy_loss
        return top_p_mask, loss

    def forward(self, A_raw, masks=None, is_training=None):
        B, H, L, _ = A_raw.shape
        device = A_raw.device
        dtype = torch.float32
        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, 0.0
        A_flat = A_raw.reshape(B * H, L, L)
        gates, loss = self.noisy_top_k_gating(A_flat, is_training) 
        gates = gates.reshape(B, H, L, -1).float()
        if masks is None:
            masks = []
            N = L // self.n_vars 
            for k in range(L):
                var_idx_k = k // N
                patch_idx_k = k % N
                S_indices = torch.arange(L, device=device)
                S_var_indices = S_indices // N
                mask_Seasonal = ((S_var_indices == var_idx_k) & (S_indices != k)).to(dtype)
                T_indices = torch.arange(L, device=device)
                T_patch_indices = T_indices % N
                mask_Spike = ((T_patch_indices == patch_idx_k) & (T_indices != k)).to(dtype)
                mask_Hetero = torch.ones(L, device=device, dtype=dtype) - mask_Seasonal - mask_Spike - torch.eye(L, device=device, dtype=dtype)[k]
                masks.append(torch.stack([mask_Seasonal, mask_Spike, mask_Hetero], dim=0))
            masks = torch.stack(masks, dim=0) 
        dynamic_mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base
        return dynamic_mask, loss


class CausalGraphConstructor(nn.Module):
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_vars = n_vars
        self.causal_query = nn.Linear(dim, dim)
        self.causal_key = nn.Linear(dim, dim)   
        self.importance_router = ImportanceAwareRouter(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, h, masks=None, alpha=0.5, is_training=False):
        B, H, L, D = h.shape
        q = self.causal_query(h)
        k = self.causal_key(h)   
        q_flat = q.reshape(B * H, L, D)
        k_flat = k.reshape(B * H, L, D)
        A = torch.bmm(q_flat, k_flat.transpose(1, 2)) 
        A = A.view(B, H, L, L) 
        A_static_sparse = A * static_sparsification(A, alpha)
        dynamic_mask, loss = self.importance_router(A_static_sparse, masks, is_training)
        A_final = A_static_sparse * dynamic_mask
        return A_final, loss 


class SemanticGraphFusionBlock(nn.Module):
    def __init__(self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_constructor = CausalGraphConstructor(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)
        self.block_aggregator = SemanticBlockAggregator(self.dim, self.n_heads)

    def forward(self, h, masks=None, alpha=0.5, is_training=False):
        B, L, D = h.shape

        h_multihead = h.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3) 
        A, router_loss = self.graph_constructor(h_multihead, masks, alpha, is_training) 
        A = torch.softmax(A, dim=-1)
        A = self.dropout(A)
        out = self.block_aggregator(A, h)
        return out, router_loss 


class SCPaTLayer(nn.Module):
    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff
        self.semantic_graph_fusion = SemanticGraphFusionBlock(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ffn_linear1 = nn.Linear(self.dim, self.d_ff)
        self.ffn_activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_linear2 = nn.Linear(self.d_ff, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, z, masks=None, alpha=0.5, is_training=False):
        res = z
        h_norm = self.norm1(z)
        delta_h, router_loss = self.semantic_graph_fusion(h_norm, masks, alpha, is_training)
        z = res + delta_h
        res = z
        h_norm = self.norm2(z)
        h_ffn = self.ffn_linear1(h_norm)
        h_ffn = self.ffn_activation(h_ffn)
        h_ffn = self.ffn_dropout(h_ffn)
        h_ffn = self.ffn_linear2(h_ffn)
        z = res + h_ffn
        
        return z, router_loss


class SCPaTEncoder(nn.Module):
    def __init__(self, hidden_dim, n_vars, d_ff=None, n_heads=4, n_blocks=3, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = hidden_dim
        self.d_ff = self.dim * 2 if d_ff is None else d_ff
        self.layers = nn.ModuleList([
            SCPaTLayer(self.dim, n_vars, self.d_ff, n_heads, top_p, dropout, in_dim)
            for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def forward(self, z, masks=None, alpha=0.5, is_training=False):
        # z: [B, L, D] (L = N_vars * Num_patches)
        total_router_loss = 0.0
        for layer in self.layers:
            z, router_loss = layer(z, masks, alpha, is_training)
            total_router_loss += router_loss
        
        avg_router_loss = total_router_loss / self.n_blocks
        return z, avg_router_loss 


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class InstanceNormalizer(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(InstanceNormalizer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm: return x
        if self.subtract_last: x = x - self.last
        else: x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm: return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last: x = x + self.last
        else: x = x + self.mean
        return x


class SemanticUnitEncoder(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.unit_projector = nn.Linear(self.patch_len, dim)
        
        self.pos = pos
        if self.pos:
            self.position_encoder = PositionalEncoder(dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.unit_projector(x)  # [B*C, L, D]
        
        if self.pos:
            x += self.position_encoder(x)
        return x


class GlobalPredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared_projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, z_encoded):
        z_forward = self.shared_projection(z_encoded) 
        z_backward = self.shared_projection(z_encoded) 
        z_out = (z_forward + z_backward) / 2.0
        return z_out

