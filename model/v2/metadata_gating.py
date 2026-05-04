import torch
import torch.nn as nn

class GatedMetadataEncoder(nn.Module):
    """Fuses Excipient Structural Encoding with Metadata via Gating."""
    def __init__(self, struct_dim=256, meta_dim=128, out_dim=256):
        super().__init__()
        
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, out_dim),
            nn.ReLU()
        )
        
        self.struct_proj = nn.Linear(struct_dim, out_dim)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, h_struct, metadata):
        h_meta = self.meta_proj(metadata)
        h_s = self.struct_proj(h_struct)
        
        # Calculate gate
        g = self.gate(torch.cat([h_s, h_meta], dim=1))
        
        # Fused output
        return g * h_s + (1 - g) * h_meta
