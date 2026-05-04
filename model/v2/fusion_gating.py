"""
STEP 4: FUSION (GATED FUSION)
Fuses API structural encoding (from MPNN) with GNN-enriched context via gating mechanism.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated Fusion: Combines structural features with GNN-enriched context.
    
    Inputs:
        - api_struct: (B, struct_dim) from MPNN structural encoding
        - context: (B, context_dim) from HGT graph encoding + metadata
    
    Output:
        - fused: (B, fusion_dim) combined representation
    
    Mechanism:
        g = sigmoid(W_g * [api_struct, context] + b_g)
        fused = g * api_struct + (1 - g) * context
    """
    
    def __init__(self, struct_dim=256, context_dim=256, fusion_dim=256, dropout=0.1):
        super().__init__()
        
        # Project structural encoding
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project context
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, api_struct, context):
        """
        Args:
            api_struct: (B, struct_dim) MPNN output for API molecule structure
            context: (B, context_dim) HGT enriched context
            
        Returns:
            fused: (B, fusion_dim) gated fusion of both modalities
        """
        # Project both inputs
        h_struct = self.struct_proj(api_struct)  # (B, fusion_dim)
        h_context = self.context_proj(context)   # (B, fusion_dim)
        
        # Calculate gate weights
        gate_input = torch.cat([h_struct, h_context], dim=1)  # (B, fusion_dim*2)
        g = self.gate(gate_input)  # (B, fusion_dim)
        
        # Weighted combination
        fused = g * h_struct + (1.0 - g) * h_context  # (B, fusion_dim)
        
        # Output projection
        fused = self.output_proj(fused)
        
        return fused


class DualModalityFusion(nn.Module):
    """
    Alternative: Multi-head fusion for richer interaction.
    Learns multiple gating patterns for different fusion modes.
    """
    
    def __init__(self, struct_dim=256, context_dim=256, fusion_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        
        assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"
        
        # Per-head projections
        self.struct_proj = nn.Linear(struct_dim, fusion_dim)
        self.context_proj = nn.Linear(context_dim, fusion_dim)
        
        # Per-head gating
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * 2, self.head_dim),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, api_struct, context):
        """
        Args:
            api_struct: (B, struct_dim)
            context: (B, context_dim)
            
        Returns:
            fused: (B, fusion_dim)
        """
        B = api_struct.shape[0]
        
        # Project to full dimension
        h_struct = self.struct_proj(api_struct)  # (B, fusion_dim)
        h_context = self.context_proj(context)   # (B, fusion_dim)
        
        # Split into heads
        h_struct_heads = h_struct.view(B, self.num_heads, self.head_dim)  # (B, heads, head_dim)
        h_context_heads = h_context.view(B, self.num_heads, self.head_dim)
        
        # Per-head gating
        fused_heads = []
        for i in range(self.num_heads):
            h_s = h_struct_heads[:, i, :]  # (B, head_dim)
            h_c = h_context_heads[:, i, :]  # (B, head_dim)
            
            gate_input = torch.cat([h_s, h_c], dim=1)  # (B, head_dim*2)
            g = self.gates[i](gate_input)  # (B, head_dim)
            
            fused_head = g * h_s + (1.0 - g) * h_c
            fused_heads.append(fused_head)
        
        # Concatenate heads
        fused = torch.cat(fused_heads, dim=1)  # (B, fusion_dim)
        
        # Output projection
        fused = self.out_proj(fused)
        
        return fused
