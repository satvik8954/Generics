"""
FULL_MODEL.py — ExciPick Heterogeneous GNN model.

Architecture:
  1. Project API features + excipient embeddings to GNN hidden dim
  2. HetGNN message passing enriches all node embeddings
  3. Per-sample: look up enriched API embedding + encode dose/route/form
  4. Score enriched context against all enriched excipient embeddings
"""

import torch
import torch.nn as nn

from model.gnn_layers import HeteroGNNEncoder
from model.api_encoder import APIProjector
from model.strength_encoder import StrengthEncoder
from model.excipient_scorer import Scorer
from config import CONFIG


class ResidualContextFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.mlp(x) + self.proj(x)

class ExciPickHGNN(nn.Module):
    def __init__(self, graph_metadata, vocab_size):
        """
        Args:
            graph_metadata: tuple from HeteroData.metadata()
                            (node_types, edge_types)
            vocab_size: number of excipients in vocabulary
        """
        super().__init__()

        # --- Node feature projectors ---
        self.api_proj = APIProjector()
        self.exc_emb = nn.Embedding(vocab_size, CONFIG["gnn_hidden"])

        # --- GNN encoder ---
        self.gnn = HeteroGNNEncoder(
            metadata=graph_metadata,
            hidden_dim=CONFIG["gnn_hidden"],
            num_layers=CONFIG["gnn_layers"],
            dropout=CONFIG["gnn_dropout"],
        )

        # --- Dose encoder (unchanged from v1) ---
        self.strength_encoder = StrengthEncoder()

        # --- Route / Form embeddings ---
        self.route_emb = nn.Embedding(CONFIG["route_vocab"], CONFIG["route_emb"])
        self.form_emb = nn.Embedding(CONFIG["form_vocab"], CONFIG["form_emb"])

        # --- Context fusion ---
        fusion_in = (
            CONFIG["gnn_hidden"]
            + CONFIG["strength_out"]
            + CONFIG["route_emb"]
            + CONFIG["form_emb"]
        )

        self.fusion = ResidualContextFusion(
            fusion_in, 
            CONFIG["context_out"], 
            CONFIG["dropout_context"]
        )

        # --- Excipient scorer ---
        self.scorer = Scorer()

    def forward(self, graph, api_idx, dose, per_unit, route, form):
        """
        Args:
            graph:    HeteroData with enriched node features
            api_idx:  (B,) long — indices of API nodes for this batch
            dose:     (B,) float — normalized dose
            per_unit: (B,) long — per-unit category ID
            route:    (B,) long — route category ID
            form:     (B,) long — dosage form category ID

        Returns:
            scores: (B, vocab_size) — raw logit scores per excipient
        """
        # 1. Project node features to GNN hidden dim
        x_dict = {
            "api": self.api_proj(graph["api"].x),
            "excipient": self.exc_emb.weight,
        }

        # 2. GNN message passing
        enriched = self.gnn(x_dict, graph.edge_index_dict)
        enriched_api = enriched["api"]           # (num_apis, gnn_hidden)
        enriched_exc = enriched["excipient"]     # (V, gnn_hidden)

        # 3. Look up this batch's API embeddings
        batch_api = enriched_api[api_idx]        # (B, gnn_hidden)

        # 4. Encode dose strength
        strength = self.strength_encoder(dose, per_unit)   # (B, strength_out)

        # 5. Encode route and form
        route_e = self.route_emb(route)          # (B, route_emb)
        form_e = self.form_emb(form)             # (B, form_emb)

        # 6. Fuse all context
        context = self.fusion(
            torch.cat([batch_api, strength, route_e, form_e], dim=1)
        )  # (B, context_out)

        # 7. Score against all enriched excipient embeddings
        scores = self.scorer(context, enriched_exc)   # (B, V)

        return scores