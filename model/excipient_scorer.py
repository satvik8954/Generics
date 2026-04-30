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

        input_dim = CONFIG["context_out"] + CONFIG["gnn_hidden"]

        self.net = nn.Sequential(
            nn.Linear(input_dim, CONFIG["scorer_hidden"]),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout_scorer"]),
            nn.Linear(CONFIG["scorer_hidden"], 1),
        )

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

        x = torch.cat([context_exp, exc_exp], dim=2)            # (B, V, input_dim)
        scores = self.net(x).squeeze(-1)                        # (B, V)

        return scores