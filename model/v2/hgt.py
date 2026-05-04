import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

class HeteroGraphTransformer(nn.Module):
    """Heterogeneous Graph Transformer (HGT) for API-EXC-EXC interactions."""
    def __init__(self, metadata, hidden_dim=256, heads=8, layers=2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    metadata=metadata, 
                    heads=heads
                )
            )
            
    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: dict of node embeddings {"api": h_api_struct, "excipient": h_exc_struct}
        edge_index_dict: dict of edges
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict
