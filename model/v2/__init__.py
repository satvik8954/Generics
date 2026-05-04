"""
ExciPick v2 Architecture Modules

Step 1: INPUT (smiles_graph.py)
  - smiles_to_graph: Convert SMILES to PyG Data with normalized atom/bond features

Step 2: STRUCTURAL ENCODING - MPNN (mpnn.py)
  - StructMPNN: Message Passing Neural Network using GINEConv
  - Returns both graph-level and node-level embeddings

Step 3: GRAPH ENCODING - HGT (hgt.py)
  - HeteroGraphTransformer: Heterogeneous Graph Transformer for API-EXC interactions
  - DualStreamHGT: Separate streams for knowledge vs similarity graphs

Step 4: FUSION - GATED FUSION (fusion_gating.py)
  - GatedFusion: Gates API structural encoding with GNN-enriched context
  - DualModalityFusion: Multi-head gating for richer fusion

Step 5: RETRIEVAL - FAISS (retrieval_faiss.py)
  - FAISSRetriever: Dense retrieval with FAISS indexing
  - HybridRetriever: Combine dense + sparse retrieval signals
  - ContextAwareRetriever: Two-stage retrieval with re-ranking

Step 6: INTERACTION MODELING (interaction_modeling.py)
  - BilinearInteraction: Simple bilinear scoring
  - MultiHeadInteraction: Multi-head attention-style interactions
  - GatedInteraction: Learned gating for selective scoring
  - InteractionHead: Ensemble of multiple interaction types
  - InteractionModule: Complete pipeline combining all approaches
"""

from .smiles_graph import smiles_to_graph, get_atom_features, get_bond_features
from .mpnn import StructMPNN
from .metadata_gating import GatedMetadataEncoder
from .hgt import HeteroGraphTransformer, DualStreamHGT
from .fusion_gating import GatedFusion, DualModalityFusion
from .retrieval_faiss import FAISSRetriever, HybridRetriever, ContextAwareRetriever
from .interaction_modeling import (
    BilinearInteraction,
    MultiHeadInteraction,
    GatedInteraction,
    InteractionHead,
    InteractionModule
)
from .siamese import SiameseNetwork, ContrastiveLoss

__all__ = [
    # Step 1: Input
    "smiles_to_graph",
    "get_atom_features",
    "get_bond_features",
    
    # Step 2: Structural Encoding
    "StructMPNN",
    "GatedMetadataEncoder",
    
    # Step 3: Graph Encoding
    "HeteroGraphTransformer",
    "DualStreamHGT",
    
    # Step 4: Fusion
    "GatedFusion",
    "DualModalityFusion",
    
    # Step 5: Retrieval
    "FAISSRetriever",
    "HybridRetriever",
    "ContextAwareRetriever",
    
    # Step 6: Interaction Modeling
    "BilinearInteraction",
    "MultiHeadInteraction",
    "GatedInteraction",
    "InteractionHead",
    "InteractionModule"
    ,
    # Step 7: Siamese
    "SiameseNetwork",
    "ContrastiveLoss"
]
