"""
STEP 9: SET TRANSFORMER & BUNDLE MODEL — Formulation Compatibility Scoring

Purpose: Score a combination (set) of excipients to predict overall formulation compatibility.

**Why Set Transformer?**
  - Excipients form an unordered set (no inherent ordering)
  - Set Transformers handle permutation-invariance naturally
  - Learns global interactions between all excipients in the set
  - Outputs single scalar: bundle compatibility score (0 to 1)

**Architecture**:
  1. Embed each excipient individually
  2. Apply Set Attention Blocks (learn pairwise interactions)
  3. Aggregate to single vector (max/mean pooling)
  4. Final MLP → Bundle Score

**Input**: Set of K excipients (variable size)
**Output**: Scalar compatibility score [0, 1]
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import math


class SetAttention(nn.Module):
    """
    Set Attention Block: Self-attention adapted for sets.
    
    Key insight: Use multi-head attention to learn pairwise relationships
    while maintaining permutation invariance through symmetric aggregation.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
            mask: (batch_size, seq_len) or (seq_len,) boolean mask (True = valid)
            
        Returns:
            (same shape as x) attention-weighted embeddings
        """
        batch_mode = x.dim() == 3
        if not batch_mode:
            x = x.unsqueeze(0)
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        Q = self.query_proj(x)  # (B, L, D)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, H, L, D_h)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, H, L, L)
        
        # Apply mask if provided
        if mask is not None:
            if not batch_mode:
                mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, L, D_h)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous()  # (B, L, H, D_h)
        out = out.view(batch_size, seq_len, hidden_dim)
        
        # Final projection
        out = self.output_proj(out)
        out = self.dropout(out)
        
        # Residual + Layer Norm
        out = self.layer_norm(out + x)
        
        if not batch_mode:
            out = out.squeeze(0)
        
        return out


class SetAttentionBlock(nn.Module):
    """
    Complete Set Attention Block: Attention + Feed-Forward.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, 
                 ff_dim: int = 512, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = SetAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
            mask: Boolean mask for valid positions
            
        Returns:
            Same shape as input
        """
        # Attention
        attn_out = self.attention(x, mask)
        
        # Feed-forward
        ff_out = self.ff(attn_out)
        out = self.layer_norm(ff_out + attn_out)
        
        return out


class SetTransformer(nn.Module):
    """
    Set Transformer: Stack of Set Attention Blocks for set encoding.
    
    Architecture:
      1. Embed individual elements (excipients)
      2. Stack K attention blocks (learn pairwise + higher-order interactions)
      3. Aggregate to single vector (permutation-invariant)
      4. Output pooled representation
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        aggregation: str = 'mean'  # 'mean', 'max', or 'mean+max'
    ):
        """
        Args:
            hidden_dim: Embedding dimension
            num_layers: Number of attention blocks
            num_heads: Number of attention heads per block
            ff_dim: Feed-forward dimension per block
            dropout: Dropout probability
            aggregation: How to aggregate set elements to single vector
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Stack of attention blocks
        self.blocks = nn.ModuleList([
            SetAttentionBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection based on aggregation
        if aggregation == 'mean+max':
            self.output_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, set_size, hidden_dim) or (set_size, hidden_dim)
            mask: (batch_size, set_size) boolean mask, or None
            
        Returns:
            (batch_size, hidden_dim) or (hidden_dim,) aggregated set representation
        """
        batch_mode = x.dim() == 3
        
        # Apply attention blocks
        out = x
        for block in self.blocks:
            out = block(out, mask)
        
        # Aggregate to single vector
        if batch_mode:
            # (B, L, D) -> (B, D)
            if self.aggregation == 'mean':
                if mask is not None:
                    # Masked mean
                    mask_expand = mask.unsqueeze(-1)  # (B, L, 1)
                    sum_out = (out * mask_expand).sum(dim=1)
                    count = mask.sum(dim=1, keepdim=True)
                    agg = sum_out / (count + 1e-8)
                else:
                    agg = out.mean(dim=1)
            elif self.aggregation == 'max':
                if mask is not None:
                    out[~mask] = float('-inf')
                agg = out.max(dim=1)[0]
            elif self.aggregation == 'mean+max':
                if self.aggregation == 'mean':
                    if mask is not None:
                        mask_expand = mask.unsqueeze(-1)
                        sum_out = (out * mask_expand).sum(dim=1)
                        count = mask.sum(dim=1, keepdim=True)
                        mean_agg = sum_out / (count + 1e-8)
                    else:
                        mean_agg = out.mean(dim=1)
                else:
                    if mask is not None:
                        out_masked = out.clone()
                        out_masked[~mask] = float('-inf')
                        mean_agg = out_masked.mean(dim=1)
                    else:
                        mean_agg = out.mean(dim=1)
                
                if mask is not None:
                    out_masked = out.clone()
                    out_masked[~mask] = float('-inf')
                    max_agg = out_masked.max(dim=1)[0]
                else:
                    max_agg = out.max(dim=1)[0]
                
                agg = torch.cat([mean_agg, max_agg], dim=1)
        else:
            # (L, D) -> (D,)
            if self.aggregation == 'mean':
                agg = out.mean(dim=0)
            elif self.aggregation == 'max':
                agg = out.max(dim=0)[0]
            elif self.aggregation == 'mean+max':
                agg = torch.cat([out.mean(dim=0), out.max(dim=0)[0]], dim=0)
        
        # Project aggregated representation
        agg = self.output_proj(agg)
        
        return agg


class SetTransformerBundleModel(nn.Module):
    """
    Complete Bundle Model: Set of Excipients → Formulation Compatibility Score
    
    Pipeline:
      1. Embed each excipient (lookup in learned embedding table)
      2. Pass through Set Transformer
      3. MLP layers to predict bundle score
      4. Output: scalar in [0, 1]
    """
    
    def __init__(
        self,
        num_excipients: int = 1299,
        excipient_emb_dim: int = 256,
        hidden_dim: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        bundle_hidden_dim: int = 128,
        max_set_size: Optional[int] = None,
        aggregation: str = 'mean'
    ):
        """
        Args:
            num_excipients: Size of excipient vocabulary
            excipient_emb_dim: Dimension of excipient embeddings
            hidden_dim: Set Transformer hidden dimension
            num_transformer_layers: Number of Set Attention blocks
            num_heads: Attention heads per block
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            bundle_hidden_dim: Hidden dimension for final MLP
            max_set_size: Maximum set size (for positional encoding if needed)
            aggregation: Aggregation method in Set Transformer
        """
        super().__init__()
        
        # Excipient embedding layer
        self.excipient_embeddings = nn.Embedding(num_excipients, excipient_emb_dim)
        
        # Optional: Project embeddings to transformer hidden dimension
        if excipient_emb_dim != hidden_dim:
            self.emb_proj = nn.Linear(excipient_emb_dim, hidden_dim)
        else:
            self.emb_proj = None
        
        # Set Transformer
        self.set_transformer = SetTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            aggregation=aggregation
        )
        
        # Bundle compatibility MLP
        self.bundle_scorer = nn.Sequential(
            nn.Linear(hidden_dim, bundle_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bundle_hidden_dim, bundle_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bundle_hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        excipient_indices: List[int] | torch.Tensor,
        batch: bool = False
    ) -> torch.Tensor:
        """
        Score a formulation bundle (combination of excipients).
        
        Args:
            excipient_indices: 
              - Single bundle: List[int] or (set_size,) tensor of excipient indices
              - Batch: (batch_size, set_size) tensor (padded)
            batch: Whether input is batched
            
        Returns:
            - Single: Scalar tensor (bundle score)
            - Batch: (batch_size,) tensor of bundle scores
        """
        if not batch:
            # Single bundle: List[int] or (set_size,)
            if isinstance(excipient_indices, list):
                excipient_indices = torch.tensor(excipient_indices, dtype=torch.long)
            
            # Embed
            x = self.excipient_embeddings(excipient_indices)  # (set_size, emb_dim)
            
            if self.emb_proj is not None:
                x = self.emb_proj(x)  # (set_size, hidden_dim)
            
            # Set Transformer (no batch dimension)
            agg = self.set_transformer(x)  # (hidden_dim,)
            
            # Score
            score = self.bundle_scorer(agg)  # scalar
            
            return score
        
        else:
            # Batch: (batch_size, set_size) padded tensor
            batch_size, set_size = excipient_indices.shape
            
            # Create mask: True where excipient_indices != 0 (assuming 0 is padding)
            mask = (excipient_indices != 0).bool()
            
            # Embed
            x = self.excipient_embeddings(excipient_indices)  # (B, L, emb_dim)
            
            if self.emb_proj is not None:
                x = self.emb_proj(x)  # (B, L, hidden_dim)
            
            # Set Transformer (with batch dimension)
            agg = self.set_transformer(x, mask)  # (B, hidden_dim)
            
            # Score
            scores = self.bundle_scorer(agg).squeeze(-1)  # (B,)
            
            return scores
    
    def score_combinations(
        self,
        combinations: List[List[int]],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Score multiple combinations at once.
        
        Args:
            combinations: List of combinations, each is List[int] of excipient indices
            device: Device to run on
            
        Returns:
            (num_combinations,) tensor of scores
        """
        # Find max set size
        max_size = max(len(combo) for combo in combinations)
        
        # Pad combinations
        padded = []
        for combo in combinations:
            padded_combo = combo + [0] * (max_size - len(combo))
            padded.append(padded_combo)
        
        excipient_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        
        # Score batch
        scores = self.forward(excipient_tensor, batch=True)
        
        return scores


class BundleModelStage9(nn.Module):
    """
    Wrapper for Stage 9: Takes beam search combinations → outputs bundle scores.
    
    Input:
      - combinations: List[List[int]] from Stage 8 (Beam Search)
      
    Output:
      - bundle_scores: (num_combinations,) tensor
      - ranked_combinations: Sorted by bundle score (descending)
    """
    
    def __init__(
        self,
        num_excipients: int = 1299,
        excipient_emb_dim: int = 256,
        hidden_dim: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.model = SetTransformerBundleModel(
            num_excipients=num_excipients,
            excipient_emb_dim=excipient_emb_dim,
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            bundle_hidden_dim=128
        )
        
        self.device = device
    
    def forward(
        self,
        combinations: List[List[int]]
    ) -> Tuple[torch.Tensor, List[Tuple[List[int], float]]]:
        """
        Args:
            combinations: From beam search stage
            
        Returns:
            bundle_scores: (num_combinations,) scores
            ranked: List of (combination, score) sorted by score descending
        """
        scores = self.model.score_combinations(combinations, device=self.device)
        
        # Rank combinations by score
        ranked_indices = torch.argsort(scores, descending=True)
        
        ranked = [
            (combinations[int(idx.item())], float(scores[idx].item()))
            for idx in ranked_indices
        ]
        
        return scores, ranked
