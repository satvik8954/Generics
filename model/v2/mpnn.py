import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch.nn import Sequential, Linear, ReLU, LayerNorm

class StructMPNN(nn.Module):
    """
    D-MPNN equivalent: Message Passing Neural Network for structural encoding.
    
    Returns both graph-level and node-level embeddings for use in downstream modules.
    
    Args:
        node_in_dim: Input dimension of node features (default 5)
        edge_in_dim: Input dimension of edge features (default 3)
        hidden_dim: Hidden dimension for MPNN layers
        depth: Number of message passing layers
    
    Returns:
        graph_emb: (graph_level_dim,) aggregated graph embedding
        node_embs: (num_nodes, hidden_dim) final node embeddings
    """
    def __init__(self, node_in_dim=5, edge_in_dim=3, hidden_dim=256, depth=4):
        super().__init__()
        
        # Normalize node features to [0, 1] range (max atomic number ~118)
        self.node_scale = 1.0 / 118.0
        
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
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
        
        # Output projections
        self.graph_readout_add = global_add_pool
        self.graph_readout_mean = global_mean_pool
        
        # Combine add + mean pooling for richer graph-level representation
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Args:
            x: (num_nodes, node_in_dim) node features
            edge_index: (2, num_edges) edge indices
            edge_attr: (num_edges, edge_in_dim) edge attributes
            batch: (num_nodes,) batch assignment (if multiple graphs)
            
        Returns:
            graph_emb: (batch_size, hidden_dim) aggregated embedding
            node_embs: (num_nodes, hidden_dim) node-level embeddings
        """
        # Project node features (with normalization)
        x = x * self.node_scale
        x = self.node_proj(x)
        
        # Message passing layers
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + x_new)  # residual + norm
        
        # Store node embeddings
        node_embs = x
        
        # Graph-level aggregation: combine add + mean pooling
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        x_add = self.graph_readout_add(x, batch)  # (batch_size, hidden_dim)
        x_mean = self.graph_readout_mean(x, batch)  # (batch_size, hidden_dim)
        
        # Combine pooling strategies
        graph_emb = self.graph_proj(torch.cat([x_add, x_mean], dim=1))  # (batch_size, hidden_dim)
        
        return graph_emb, node_embs
