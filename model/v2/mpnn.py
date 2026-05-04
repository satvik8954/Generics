import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, LayerNorm

class StructMPNN(nn.Module):
    """D-MPNN equivalent: Message Passing Neural Network for structural encoding."""
    def __init__(self, node_in_dim=5, edge_in_dim=3, hidden_dim=256, depth=4):
        super().__init__()
        
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(depth):
            # GINEConv handles edge attributes
            conv = GINEConv(
                nn=Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_in_dim
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_dim))
            
        self.readout = global_add_pool
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_proj(x)
        
        for conv, norm in zip(self.convs, self.norms):
            # GINEConv uses edge attributes
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + x_new)  # residual + norm
            
        # Aggregate node features into a graph-level embedding
        return self.readout(x, batch)
