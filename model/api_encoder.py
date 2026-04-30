"""
api_encoder.py — API feature projector for ExciPick HetGNN.

Projects 20 molecular descriptors to GNN hidden dimension.
The heavy representational lifting is done by the GNN layers.
"""

import torch.nn as nn
from config import CONFIG


class APIProjector(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(CONFIG["api_in"], CONFIG["gnn_hidden"]),
            nn.LayerNorm(CONFIG["gnn_hidden"]),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)