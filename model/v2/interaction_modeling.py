"""
STEP 6: INTERACTION MODELING
Computes rich interactions between fused context and retrieved excipient candidates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearInteraction(nn.Module):
    """
    Bilinear interaction layer for context-excipient scoring.
    
    Score(context, exc) = context^T * W * exc + b
    
    Efficient implementation for batch scoring.
    """
    
    def __init__(self, context_dim=256, exc_dim=256, out_dim=1):
        super().__init__()
        
        self.bilinear = nn.Bilinear(context_dim, exc_dim, out_dim)
        
    def forward(self, context, exc_embs):
        """
        Args:
            context: (B, context_dim)
            exc_embs: (B, K, exc_dim) retrieved excipient embeddings
            
        Returns:
            scores: (B, K, out_dim) interaction scores
        """
        B, K = exc_embs.shape[0], exc_embs.shape[1]
        
        # Expand context to match excipient batch
        context_exp = context.unsqueeze(1).expand(-1, K, -1)  # (B, K, context_dim)
        
        # Reshape for bilinear
        context_flat = context_exp.reshape(B * K, -1)  # (B*K, context_dim)
        exc_flat = exc_embs.reshape(B * K, -1)  # (B*K, exc_dim)
        
        # Compute scores
        scores = self.bilinear(context_flat, exc_flat)  # (B*K, out_dim)
        scores = scores.view(B, K, -1)  # (B, K, out_dim)
        
        return scores


class MultiHeadInteraction(nn.Module):
    """
    Multi-head attention-style interaction between context and excipients.
    
    Each head learns different interaction patterns (e.g., hydrophobicity,
    solubility, thermal stability, etc.).
    """
    
    def __init__(self, context_dim=256, exc_dim=256, num_heads=8, out_dim=1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = max(context_dim // num_heads, 32)
        
        # Per-head projections
        self.context_proj = nn.Linear(context_dim, num_heads * self.head_dim)
        self.exc_proj = nn.Linear(exc_dim, num_heads * self.head_dim)
        
        # Per-head scoring (bilinear)
        self.head_scores = nn.ModuleList([
            nn.Bilinear(self.head_dim, self.head_dim, 1)
            for _ in range(num_heads)
        ])
        
        # Combine head scores
        self.head_combine = nn.Sequential(
            nn.Linear(num_heads, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, out_dim)
        )
        
    def forward(self, context, exc_embs):
        """
        Args:
            context: (B, context_dim)
            exc_embs: (B, K, exc_dim)
            
        Returns:
            scores: (B, K, out_dim)
        """
        B, K = exc_embs.shape[0], exc_embs.shape[1]
        
        # Project to multi-head space
        context_proj = self.context_proj(context)  # (B, num_heads * head_dim)
        exc_proj = self.exc_proj(exc_embs)  # (B, K, num_heads * head_dim)
        
        # Reshape into heads
        context_heads = context_proj.view(B, self.num_heads, self.head_dim)  # (B, heads, head_dim)
        exc_heads = exc_proj.view(B, K, self.num_heads, self.head_dim)  # (B, K, heads, head_dim)
        
        # Per-head scoring
        head_scores_list = []
        for h in range(self.num_heads):
            ctx_h = context_heads[:, h, :]  # (B, head_dim)
            exc_h = exc_heads[:, :, h, :]  # (B, K, head_dim)
            
            # Expand context for all K candidates
            ctx_h_exp = ctx_h.unsqueeze(1).expand(-1, K, -1)  # (B, K, head_dim)
            ctx_h_flat = ctx_h_exp.reshape(B * K, -1)
            exc_h_flat = exc_h.reshape(B * K, -1)
            
            h_score = self.head_scores[h](ctx_h_flat, exc_h_flat)  # (B*K, 1)
            h_score = h_score.view(B, K)  # (B, K)
            head_scores_list.append(h_score)
        
        # Stack head scores
        all_head_scores = torch.stack(head_scores_list, dim=2)  # (B, K, num_heads)
        
        # Combine heads
        scores = self.head_combine(all_head_scores)  # (B, K, out_dim)
        
        return scores


class GatedInteraction(nn.Module):
    """
    Gated interaction mechanism: learns what aspects of excipients are relevant.
    
    Gate(context, exc) = sigmoid(W_g * [context, exc] + b_g)
    Score = Gate * Bilinear(context, exc)
    """
    
    def __init__(self, context_dim=256, exc_dim=256, out_dim=1, dropout=0.1):
        super().__init__()
        
        # Interaction scoring
        self.interaction = nn.Bilinear(context_dim, exc_dim, out_dim)
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim + exc_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, context, exc_embs):
        """
        Args:
            context: (B, context_dim)
            exc_embs: (B, K, exc_dim)
            
        Returns:
            scores: (B, K, out_dim) gated interaction scores
        """
        B, K = exc_embs.shape[0], exc_embs.shape[1]
        
        # Expand context
        context_exp = context.unsqueeze(1).expand(-1, K, -1)  # (B, K, context_dim)
        
        # Concatenate for gating
        combined = torch.cat([context_exp, exc_embs], dim=2)  # (B, K, context_dim + exc_dim)
        combined_flat = combined.reshape(B * K, -1)
        
        # Gate scores
        gate = self.gate_net(combined_flat).view(B, K, -1)  # (B, K, out_dim)
        
        # Interaction scores
        context_flat = context_exp.reshape(B * K, -1)
        exc_flat = exc_embs.reshape(B * K, -1)
        interaction = self.interaction(context_flat, exc_flat).view(B, K, -1)  # (B, K, out_dim)
        
        # Gated combination
        scores = gate * interaction
        
        return scores


class InteractionHead(nn.Module):
    """
    Full interaction head combining multiple scoring mechanisms.
    
    Ensembles multiple interaction types:
    - Bilinear interaction
    - Multi-head attention interaction
    - Gated interaction
    """
    
    def __init__(self, context_dim=256, exc_dim=256, num_heads=8, 
                 use_bilinear=True, use_multihead=True, use_gated=True, dropout=0.1):
        super().__init__()
        
        self.use_bilinear = use_bilinear
        self.use_multihead = use_multihead
        self.use_gated = use_gated
        
        num_models = sum([use_bilinear, use_multihead, use_gated])
        
        if use_bilinear:
            self.bilinear = BilinearInteraction(context_dim, exc_dim, 1)
        if use_multihead:
            self.multihead = MultiHeadInteraction(context_dim, exc_dim, num_heads, 1)
        if use_gated:
            self.gated = GatedInteraction(context_dim, exc_dim, 1, dropout)
        
        # Ensemble combiner (learned weighted average)
        self.ensemble_weight = nn.Sequential(
            nn.Linear(num_models, num_models),
            nn.ReLU(),
            nn.Linear(num_models, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, context, exc_embs):
        """
        Args:
            context: (B, context_dim)
            exc_embs: (B, K, exc_dim)
            
        Returns:
            scores: (B, K, 1) ensemble interaction scores
        """
        B, K = exc_embs.shape[0], exc_embs.shape[1]
        scores_list = []
        
        if self.use_bilinear:
            scores_list.append(self.bilinear(context, exc_embs))  # (B, K, 1)
        if self.use_multihead:
            scores_list.append(self.multihead(context, exc_embs))  # (B, K, 1)
        if self.use_gated:
            scores_list.append(self.gated(context, exc_embs))  # (B, K, 1)
        
        # Stack and compute ensemble weights
        scores_stacked = torch.cat(scores_list, dim=2)  # (B, K, num_models)
        
        # Learn per-sample ensemble weights
        scores_flat = scores_stacked.reshape(B * K, -1)
        weights = self.ensemble_weight(scores_flat)  # (B*K, num_models)
        
        # Weighted combination
        weighted_scores = (scores_stacked.reshape(B * K, -1) * weights).sum(dim=1, keepdim=True)
        weighted_scores = weighted_scores.view(B, K, 1)  # (B, K, 1)
        
        return weighted_scores


class InteractionModule(nn.Module):
    """
    Complete interaction modeling pipeline.
    
    Inputs:
        - context: (B, context_dim) fused context from step 4
        - retrieved_indices: (B, K) indices of candidate excipients
        - exc_embeddings: (V, exc_dim) all excipient embeddings
        
    Output:
        - scores: (B, K) interaction scores for each candidate
    """
    
    def __init__(self, context_dim=256, exc_dim=256, num_heads=8, 
                 ensemble_mode='weighted', dropout=0.1):
        super().__init__()
        
        self.ensemble_mode = ensemble_mode
        
        if ensemble_mode == 'weighted':
            self.scorer = InteractionHead(
                context_dim, exc_dim, num_heads,
                use_bilinear=True, use_multihead=True, use_gated=True,
                dropout=dropout
            )
        elif ensemble_mode == 'multihead':
            self.scorer = MultiHeadInteraction(context_dim, exc_dim, num_heads, 1)
        elif ensemble_mode == 'gated':
            self.scorer = GatedInteraction(context_dim, exc_dim, 1, dropout)
        else:  # bilinear
            self.scorer = BilinearInteraction(context_dim, exc_dim, 1)
    
    def forward(self, context, retrieved_indices, exc_embeddings):
        """
        Args:
            context: (B, context_dim)
            retrieved_indices: (B, K) LongTensor of excipient indices
            exc_embeddings: (V, exc_dim) all excipient embeddings
            
        Returns:
            scores: (B, K) interaction scores
        """
        B, K = retrieved_indices.shape
        
        # Gather embeddings for retrieved candidates
        exc_embs = exc_embeddings[retrieved_indices]  # (B, K, exc_dim)
        
        # Compute interaction scores
        scores = self.scorer(context, exc_embs)  # (B, K, 1)
        scores = scores.squeeze(2)  # (B, K)
        
        return scores
