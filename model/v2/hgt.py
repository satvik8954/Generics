import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_mean_pool


class HeteroGraphTransformer(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for API-EXC-EXC interactions.
    
    Enriches API and Excipient embeddings by learning from heterogeneous graph structure:
    - API nodes connected to Excipient nodes (knowledge graph)
    - Excipient nodes connected to each other (similarity graph)
    
    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        hidden_dim: Embedding dimension
        heads: Number of attention heads
        layers: Number of HGT layers
        
    Input:
        x_dict: Dict mapping node types to embeddings
                {"api": (num_apis, hidden_dim), "excipient": (num_excs, hidden_dim)}
        edge_index_dict: Dict mapping edge types to edge indices
    
    Output:
        x_dict: Updated node embeddings enriched by graph structure
    """
    
    def __init__(self, metadata, hidden_dim=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.metadata = metadata
        
        # HGT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Dropout layer (applied after each HGT conv)
        self.dropout = nn.Dropout(dropout)
        
        # Optional: learnable node-type embeddings for better differentiation
        node_types = metadata[0]
        self.node_type_emb = nn.ParameterDict({
            node_type: nn.Parameter(torch.randn(1, hidden_dim))
            for node_type in node_types
        })
        
    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dict of node embeddings {"api": ..., "excipient": ...}
            edge_index_dict: Dict of edge indices for each edge type
                            e.g., {"api_exc": edge_idx, "exc_exc": edge_idx}
        
        Returns:
            x_dict: Updated node embeddings after GNN layers
        """
        # Add node-type specific information
        for node_type in x_dict.keys():
            x_dict[node_type] = x_dict[node_type] + self.node_type_emb[node_type]
        
        # Apply HGT layers
        for conv, norm in zip(self.convs, self.norms):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply normalization, dropout, and residual connection per node type
            for node_type in x_dict.keys():
                if node_type in x_dict_new:
                    x_dict[node_type] = norm(self.dropout(x_dict[node_type] + x_dict_new[node_type]))
                else:
                    x_dict[node_type] = norm(self.dropout(x_dict[node_type]))
        
        return x_dict
    
    def get_node_embedding(self, x_dict, node_type):
        """Extract embeddings for a specific node type."""
        return x_dict[node_type]


class DualStreamHGT(nn.Module):
    """
    Dual-stream HGT: Separate encoding streams for different edge types.
    
    Streams:
        1. Structure stream: API-EXC knowledge graph
        2. Similarity stream: EXC-EXC similarity graph
    
    Allows independent modeling of knowledge vs. similarity relationships.
    """
    
    def __init__(self, metadata, hidden_dim=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.metadata = metadata
        
        # Structure stream (API-EXC edges)
        self.struct_convs = nn.ModuleList()
        self.struct_norms = nn.ModuleList()
        
        # Similarity stream (EXC-EXC edges)
        self.sim_convs = nn.ModuleList()
        self.sim_norms = nn.ModuleList()
        
        for _ in range(layers):
            # Structure stream
            struct_conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads
            )
            self.struct_convs.append(struct_conv)
            self.struct_norms.append(nn.LayerNorm(hidden_dim))
            
            # Similarity stream
            sim_conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads
            )
            self.sim_convs.append(sim_conv)
            self.sim_norms.append(nn.LayerNorm(hidden_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_dict, struct_edge_dict, sim_edge_dict):
        """
        Args:
            x_dict: Initial node embeddings
            struct_edge_dict: Structure stream edge indices (API-EXC)
            sim_edge_dict: Similarity stream edge indices (EXC-EXC)
        
        Returns:
            x_dict: Fused embeddings
        """
        # Structure stream
        x_struct = dict(x_dict)
        for conv, norm in zip(self.struct_convs, self.struct_norms):
            x_struct_new = conv(x_struct, struct_edge_dict)
            for node_type in x_struct.keys():
                if node_type in x_struct_new:
                    x_struct[node_type] = norm(self.dropout(x_struct[node_type] + x_struct_new[node_type]))
                else:
                    x_struct[node_type] = norm(self.dropout(x_struct[node_type]))
        
        # Similarity stream
        x_sim = dict(x_dict)
        for conv, norm in zip(self.sim_convs, self.sim_norms):
            x_sim_new = conv(x_sim, sim_edge_dict)
            for node_type in x_sim.keys():
                if node_type in x_sim_new:
                    x_sim[node_type] = norm(self.dropout(x_sim[node_type] + x_sim_new[node_type]))
                else:
                    x_sim[node_type] = norm(self.dropout(x_sim[node_type]))
        
        # Fuse streams with gating
        x_fused = {}
        for node_type in x_struct.keys():
            # Compute fusion gate
            combined = torch.cat([x_struct[node_type], x_sim[node_type]], dim=1)
            gate = self.fusion_gate(combined)
            
            # Weighted combination
            x_fused[node_type] = gate * x_struct[node_type] + (1.0 - gate) * x_sim[node_type]
        
        return x_fused
