"""
excipient_scorer.py — Scores each excipient against a formulation context.

In the HetGNN version, excipient embeddings come from the GNN encoder
(enriched via message passing), not from a local nn.Embedding.
"""

import torch
import torch.nn as nn
from config import CONFIG


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()

        # Bilinear interaction: context (192) x excipient (128) -> 1 logit
        self.bilinear = nn.Bilinear(CONFIG["context_out"], CONFIG["gnn_hidden"], 1)

    def forward(self, context, exc_embs):
        """
        Args:
            context:  (B, context_out) — fused formulation context
            exc_embs: (V, gnn_hidden)  — GNN-enriched excipient embeddings

        Returns:
            scores: (B, V) — one logit per excipient
        """
        B = context.shape[0]
        V = exc_embs.shape[0]

        # Tile context and excipient embeddings for pairwise scoring
        context_exp = context.unsqueeze(1).expand(-1, V, -1)   # (B, V, context_out)
        exc_exp = exc_embs.unsqueeze(0).expand(B, -1, -1)      # (B, V, gnn_hidden)

        # Bilinear expects (B*V, in1), (B*V, in2)
        scores = self.bilinear(context_exp.reshape(-1, CONFIG["context_out"]), 
                               exc_exp.reshape(-1, CONFIG["gnn_hidden"]))
        return scores.view(B, V)