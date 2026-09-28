import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SCPaT_layers import InstanceNormalizer, SemanticUnitEncoder, SCPaTEncoder, GlobalPredictionHead  

class Model(nn.Module): 
    def __init__(self, configs):
        super().__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = configs.stride if hasattr(configs, 'stride') and configs.stride is not None else self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  
        self.alpha = 0.1 if configs.alpha is None else configs.alpha 
        self.top_p = 0.5 if configs.top_p is None else configs.top_p 
        self.use_RevIN = True 
        self.norm_layer = InstanceNormalizer(configs.enc_in, affine=self.use_RevIN, subtract_last=False)
        self.semantic_unit_encoder = SemanticUnitEncoder(self.dim, self.patch_len, self.stride, configs.pos)
        self.total_patches = self.n_vars * self.num_patches # N_vars * L
        self.spformer_encoder = SCPaTEncoder(self.dim, self.n_vars, self.d_ff,
                                                configs.n_heads, configs.e_layers, self.top_p, configs.dropout, 
                                                in_dim=self.total_patches)
        self.prediction_head = GlobalPredictionHead(
            input_dim=self.dim * self.num_patches, 
            output_dim=self.pred_len
        )
        assert configs.enc_in == self.n_vars, "configs.enc_in must equal configs.c_out (n_vars)"


    def forward(self, z_in, masks, is_training=False, target=None):
        B, T_in, N_vars = z_in.shape

        z = self.norm_layer(z_in, 'norm')  

        z = z.permute(0, 2, 1)      
        z = z.reshape(B * N_vars, T_in) 
        z = self.semantic_unit_encoder(z)   
        
        z = z.reshape(B, N_vars, self.num_patches, self.dim)
        z = z.reshape(B, N_vars * self.num_patches, self.dim)
        
        z_encoded, router_loss = self.spformer_encoder(z, masks, self.alpha, is_training)

        z_decoded = z_encoded.reshape(B, self.n_vars, self.num_patches, self.dim) 
        z_decoded = z_decoded.flatten(start_dim=2)
        z_out = self.prediction_head(z_decoded) 

        z_out = z_out.permute(0, 2, 1)
        z_out = self.norm_layer(z_out, 'denorm')

        return z_out, router_loss