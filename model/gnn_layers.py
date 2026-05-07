"""
gnn_layers.py — Heterogeneous GNN encoder for ExciPick.

Uses PyG's HeteroConv with SAGEConv per edge type.
Each layer: HeteroConv → LayerNorm → ReLU → Dropout (+ residual).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNNEncoder(nn.Module):
    """
    Multi-layer heterogeneous GNN using SAGEConv per edge type.

    Args:
        metadata: tuple of (node_types, edge_types) from HeteroData.metadata()
        hidden_dim: hidden dimension for all layers
        num_layers: number of message passing rounds
        dropout: dropout rate
    """

    def __init__(self, metadata, hidden_dim, num_layers, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # One SAGEConv per edge type
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            # LayerNorm per node type
            norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in metadata[0]
            })
            self.norms.append(norm_dict)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: {node_type: (num_nodes, hidden_dim)} node features
            edge_index_dict: {edge_type: (2, num_edges)} edge indices

        Returns:
            x_dict: enriched node embeddings, same structure as input
        """
        for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Message passing
            x_dict_new = conv(x_dict, edge_index_dict)

            if layer_idx == 0:
                # No residual — input dim != hidden_dim
                x_dict = {
                    key: F.dropout(
                        F.relu(norm_dict[key](x_dict_new[key])),
                        p=self.dropout,
                        training=self.training,
                    )
                    for key in x_dict_new.keys()
                }
            else:
                # Residual safe from layer 1 onward
                x_dict = {
                    key: F.dropout(
                        F.relu(norm_dict[key](x_dict_new[key] + x_dict[key])),
                        p=self.dropout,
                        training=self.training,
                    )
                    for key in x_dict_new.keys()
                }

        return x_dict
