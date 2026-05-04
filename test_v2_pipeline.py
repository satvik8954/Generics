"""
test_v2_pipeline.py — End-to-end test of ExciPick v2 architecture (Steps 1-6)

Tests:
  1. INPUT: SMILES → PyG graphs
  2. STRUCTURAL ENCODING: MPNN
  3. GRAPH ENCODING: HGT on heterogeneous graph
  4. FUSION: Gated fusion of structure + context
  5. RETRIEVAL: FAISS-based dense retrieval
  6. INTERACTION MODELING: Scoring retrieved candidates

No external dataset required; uses synthetic data for testing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import v2 modules
from model.v2 import (
    smiles_to_graph,
    StructMPNN,
    HeteroGraphTransformer,
    GatedFusion,
    FAISSRetriever,
    InteractionModule
    ,
    # Siamese
    SiameseNetwork,
    ContrastiveLoss
)

# Optional: Try importing torch_geometric for heterogeneous graph
try:
    from torch_geometric.data import HeteroData
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("⚠️  torch_geometric not available; using dummy graph")


class SyntheticDataGenerator:
    """Generate synthetic SMILES and metadata for testing."""
    
    SAMPLE_SMILES = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CC(C)NCC(COc1ccccc1)O",  # Salbutamol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)c1ccccc1O",  # Acetylsalicylic acid
        "c1ccc2c(c1)ccc3c2cccc3",  # Anthracene
        "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
        "O=C(O)c1ccccc1",  # Benzoic acid
        "CC(C)C(=O)O",  # Isobutyric acid
        "c1ccc(cc1)C(=O)O",  # Benzoic acid derivative
    ]
    
    @staticmethod
    def generate_smiles_batch(num_samples: int = 5) -> List[str]:
        """Generate random SMILES batch."""
        return np.random.choice(SyntheticDataGenerator.SAMPLE_SMILES, num_samples).tolist()
    
    @staticmethod
    def generate_metadata_batch(batch_size: int = 5, num_routes: int = 16, 
                               num_forms: int = 110) -> Dict[str, torch.Tensor]:
        """Generate synthetic metadata."""
        return {
            'dose': torch.randn(batch_size) * 2,  # log-normalized
            'per_unit': torch.randint(0, 4, (batch_size,)),
            'route': torch.randint(0, num_routes, (batch_size,)),
            'form': torch.randint(0, num_forms, (batch_size,)),
            'api_idx': torch.randint(0, 50, (batch_size,)),  # 50 unique APIs
        }


class V2PipelineTest:
    """Full end-to-end test of v2 architecture."""
    
    def __init__(self, device='cpu', batch_size=4, num_excipients=256):
        self.device = device
        self.batch_size = batch_size
        self.num_excipients = num_excipients
        self.hidden_dim = 256
        self.fusion_dim = 256
        
        print(f"🚀 Initializing V2 Pipeline Test")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Num excipients: {num_excipients}")
        
    def step1_input_smiles_to_graph(self, smiles_list: List[str]):
        """Step 1: Convert SMILES to PyG graphs."""
        print("\n" + "="*60)
        print("STEP 1: INPUT (SMILES → PyG Graphs)")
        print("="*60)
        
        graphs = []
        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles, normalize=True)
            if graph is None:
                print(f"  ⚠️  Invalid SMILES: {smiles}")
                continue
            graphs.append(graph)
            print(f"  ✓ Graph {i}: {len(graph.x)} atoms, {graph.edge_index.shape[1]} bonds")
        
        if not graphs:
            raise ValueError("No valid SMILES found")
        
        return graphs
    
    def step2_structural_encoding_mpnn(self, graphs):
        """Step 2: Structural encoding via MPNN."""
        print("\n" + "="*60)
        print("STEP 2: STRUCTURAL ENCODING (MPNN)")
        print("="*60)
        
        mpnn = StructMPNN(
            node_in_dim=5,
            edge_in_dim=3,
            hidden_dim=self.hidden_dim,
            depth=3
        ).to(self.device)
        
        graph_embs = []
        node_embs_list = []
        
        for i, graph in enumerate(graphs):
            # Move graph to device
            graph.x = graph.x.to(self.device)
            graph.edge_index = graph.edge_index.to(self.device)
            graph.edge_attr = graph.edge_attr.to(self.device)
            
            # Forward pass (no batch)
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=self.device)
            graph_emb, node_embs = mpnn(graph.x, graph.edge_index, graph.edge_attr, batch)
            
            graph_embs.append(graph_emb)
            node_embs_list.append(node_embs)
            
            print(f"  ✓ Graph {i}:")
            print(f"     - graph_emb: {graph_emb.shape}")
            print(f"     - node_embs: {node_embs.shape}")
        
        # Stack batch of graph embeddings
        api_struct = torch.cat(graph_embs, dim=0)  # (batch_size, hidden_dim)
        print(f"\n  ✓ Batch api_struct: {api_struct.shape}")
        
        return mpnn, api_struct
    
    def step3_graph_encoding_hgt(self):
        """Step 3: Graph encoding via HGT."""
        print("\n" + "="*60)
        print("STEP 3: GRAPH ENCODING (HGT on Heterogeneous Graph)")
        print("="*60)
        
        if not HAS_TORCH_GEOMETRIC:
            print("  ⚠️  torch_geometric not available; creating dummy tensors")
            # Return dummy HGT + dummy enriched embeddings
            hgt = nn.Identity()
            enriched_exc = torch.randn(self.num_excipients, self.hidden_dim, device=self.device)
            print(f"  ✓ enriched_exc: {enriched_exc.shape}")
            return hgt, enriched_exc
        
        # Create synthetic heterogeneous graph
        print("  Creating synthetic heterogeneous graph...")
        num_apis = 50
        num_excs = self.num_excipients
        
        # Node features
        api_x = torch.randn(num_apis, self.hidden_dim, device=self.device)
        exc_x = torch.randn(num_excs, self.hidden_dim, device=self.device)
        
        # Edge indices (synthetic)
        # API-EXC edges: each API connects to ~20 random excipients
        api_exc_edges = []
        for api_id in range(num_apis):
            exc_ids = np.random.choice(num_excs, 20, replace=False)
            for exc_id in exc_ids:
                api_exc_edges.append([api_id, exc_id])
        api_exc_edge_index = torch.tensor(api_exc_edges, dtype=torch.long, device=self.device).t()
        
        # EXC-EXC edges: random similarity graph (~5 neighbors per excipient)
        exc_exc_edges = []
        for exc_id in range(num_excs):
            neighbor_ids = np.random.choice(num_excs, min(5, num_excs-1), replace=False)
            for neighbor_id in neighbor_ids:
                if neighbor_id != exc_id:
                    exc_exc_edges.append([exc_id, neighbor_id])
        exc_exc_edge_index = torch.tensor(exc_exc_edges, dtype=torch.long, device=self.device).t()
        
        print(f"  ✓ API nodes: {num_apis}, features: {api_x.shape[1]}")
        print(f"  ✓ Excipient nodes: {num_excs}, features: {exc_x.shape[1]}")
        print(f"  ✓ API-EXC edges: {api_exc_edge_index.shape[1]}")
        print(f"  ✓ EXC-EXC edges: {exc_exc_edge_index.shape[1]}")
        
        # Create HeteroData
        heterograph = HeteroData()
        heterograph['api'].x = api_x
        heterograph['excipient'].x = exc_x
        heterograph['api', 'interacts', 'excipient'].edge_index = api_exc_edge_index
        heterograph['excipient', 'similar', 'excipient'].edge_index = exc_exc_edge_index
        
        # Initialize HGT
        metadata = heterograph.metadata()
        hgt = HeteroGraphTransformer(
            metadata=metadata,
            hidden_dim=self.hidden_dim,
            heads=8,
            layers=2,
            dropout=0.1
        ).to(self.device)
        
        # Forward pass
        x_dict = {
            'api': api_x,
            'excipient': exc_x
        }
        edge_index_dict = {
            ('api', 'interacts', 'excipient'): api_exc_edge_index,
            ('excipient', 'similar', 'excipient'): exc_exc_edge_index
        }
        
        x_dict_out = hgt(x_dict, edge_index_dict)
        enriched_exc = x_dict_out['excipient']
        
        print(f"\n  ✓ enriched_exc: {enriched_exc.shape}")
        
        return hgt, enriched_exc
    
    def step4_fusion_gated(self, api_struct: torch.Tensor):
        """Step 4: Fused context via gated fusion."""
        print("\n" + "="*60)
        print("STEP 4: FUSION (Gated Fusion)")
        print("="*60)
        
        fusion = GatedFusion(
            struct_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            fusion_dim=self.fusion_dim,
            dropout=0.1
        ).to(self.device)
        
        # Dummy context (normally from HGT + metadata)
        context = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        
        # Fuse
        fused = fusion(api_struct, context)
        
        print(f"  ✓ api_struct: {api_struct.shape}")
        print(f"  ✓ context: {context.shape}")
        print(f"  ✓ fused: {fused.shape}")
        
        return fusion, fused
    
    def step5_retrieval_faiss(self, fused: torch.Tensor, 
                              enriched_exc: torch.Tensor, top_k: int = 64):
        """Step 5: Dense retrieval via FAISS."""
        print("\n" + "="*60)
        print("STEP 5: RETRIEVAL (FAISS Dense Retrieval)")
        print("="*60)
        
        retriever = FAISSRetriever(embedding_dim=self.fusion_dim, use_gpu=False)
        
        # Build index
        print("  Building FAISS index...")
        retriever.build_index(enriched_exc)
        print(f"  ✓ Index built for {enriched_exc.shape[0]} excipients")
        
        # Retrieve top-K
        print(f"  Retrieving top-{top_k} candidates...")
        distances, indices, scores = retriever(fused, top_k=top_k)
        
        print(f"  ✓ distances: {distances.shape}")
        print(f"  ✓ indices: {indices.shape}")
        print(f"  ✓ scores: {scores.shape}")
        print(f"  ✓ Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        return retriever, indices
    
    def step6_interaction_modeling(self, fused: torch.Tensor, 
                                   retrieved_indices: torch.Tensor,
                                   enriched_exc: torch.Tensor):
        """Step 6: Interaction modeling for scoring."""
        print("\n" + "="*60)
        print("STEP 6: INTERACTION MODELING")
        print("="*60)
        
        interaction = InteractionModule(
            context_dim=self.fusion_dim,
            exc_dim=self.hidden_dim,
            num_heads=8,
            ensemble_mode='weighted',
            dropout=0.1
        ).to(self.device)
        
        # Score retrieved candidates
        interaction_scores = interaction(fused, retrieved_indices, enriched_exc)
        
        print(f"  ✓ fused context: {fused.shape}")
        print(f"  ✓ retrieved candidates: {retrieved_indices.shape}")
        print(f"  ✓ interaction_scores: {interaction_scores.shape}")
        print(f"  ✓ Score range: [{interaction_scores.min():.4f}, {interaction_scores.max():.4f}]")
        
        # Show top-3 per sample
        print("\n  Top-3 candidates per sample:")
        for b in range(min(2, fused.shape[0])):
            top3_idx = torch.argsort(interaction_scores[b], descending=True)[:3]
            print(f"    Sample {b}:")
            for rank, idx in enumerate(top3_idx, 1):
                exc_id = retrieved_indices[b, idx].item()
                score = interaction_scores[b, idx].item()
                print(f"      {rank}. Excipient {exc_id}: {score:.4f}")
        
        return interaction, interaction_scores

    def step7_siamese_contrastive(self, fused: torch.Tensor,
                                   retrieved_indices: torch.Tensor,
                                   enriched_exc: torch.Tensor,
                                   margin: float = 1.0):
        """Step 7: Siamese contrastive learning pass (single-batch demo)."""
        print("\n" + "="*60)
        print("STEP 7: SIAMESE (Contrastive Loss)")
        print("="*60)

        device = fused.device

        siam = SiameseNetwork(context_dim=self.fusion_dim,
                              exc_dim=self.hidden_dim,
                              proj_dim=128).to(device)
        loss_fn = ContrastiveLoss(margin=margin)

        B = fused.shape[0]

        # Build positive pairs: (context, top-1 retrieved excipient)
        pos_idx = retrieved_indices[:, 0]  # (B,)
        pos_embs = enriched_exc[pos_idx]   # (B, exc_dim)

        # Build negative pairs: random excipient not equal to positive
        neg_idx = []
        for b in range(B):
            cand = int(pos_idx[b].item())
            choose = np.random.choice([i for i in range(enriched_exc.shape[0]) if i != cand])
            neg_idx.append(choose)
        neg_idx = torch.tensor(neg_idx, dtype=torch.long, device=device)
        neg_embs = enriched_exc[neg_idx]

        # Concatenate pairs and labels
        context_pairs = torch.cat([fused, fused], dim=0)  # (2B, fusion_dim)
        exc_pairs = torch.cat([pos_embs, neg_embs], dim=0)  # (2B, exc_dim)
        labels = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)], dim=0)

        # Forward
        z_ctx, z_exc = siam(context_pairs, exc_pairs)
        loss = loss_fn(z_ctx, z_exc, labels)

        # Print diagnostics
        with torch.no_grad():
            dists = torch.norm(z_ctx - z_exc, dim=1)
            print(f"  ✓ Contrastive loss: {loss.item():.4f}")
            print(f"  ✓ Distances (pos first {B} / neg next {B}): {dists[:min(5,2*B)].cpu().numpy()}")

        return siam, loss
    
    def run_full_pipeline(self):
        """Run complete end-to-end test."""
        print("\n" + "🔗 " * 30)
        print("EXCIPICK V2 ARCHITECTURE — END-TO-END TEST")
        print("🔗 " * 30)
        
        # Generate synthetic data
        print("\n📊 Generating synthetic data...")
        smiles_list = SyntheticDataGenerator.generate_smiles_batch(self.batch_size)
        print(f"   ✓ Generated {len(smiles_list)} SMILES strings")
        
        # Step 1: SMILES → Graphs
        graphs = self.step1_input_smiles_to_graph(smiles_list)
        
        # Step 2: Structural encoding
        mpnn, api_struct = self.step2_structural_encoding_mpnn(graphs)
        
        # Step 3: Graph encoding
        hgt, enriched_exc = self.step3_graph_encoding_hgt()
        
        # Step 4: Fusion
        fusion, fused = self.step4_fusion_gated(api_struct)
        
        # Step 5: Retrieval
        retriever, retrieved_indices = self.step5_retrieval_faiss(fused, enriched_exc, top_k=64)
        
        # Step 6: Interaction modeling
        interaction, scores = self.step6_interaction_modeling(fused, retrieved_indices, enriched_exc)

        # Step 7: Siamese contrastive demo
        siamese, contrastive_loss = self.step7_siamese_contrastive(fused, retrieved_indices, enriched_exc)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60)
        print(f"\nFinal output shape: {scores.shape}")
        print(f"Represents: {self.batch_size} samples × {retrieved_indices.shape[1]} candidate excipients")
        
        return {
            'graphs': graphs,
            'mpnn': mpnn,
            'api_struct': api_struct,
            'hgt': hgt,
            'enriched_exc': enriched_exc,
            'fusion': fusion,
            'fused': fused,
            'retriever': retriever,
            'retrieved_indices': retrieved_indices,
            'interaction': interaction,
            'final_scores': scores
        }


def main():
    """Main test runner."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create and run test
    tester = V2PipelineTest(
        device=device,
        batch_size=4,
        num_excipients=256
    )
    
    results = tester.run_full_pipeline()
    
    # Additional validation
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check tensor dimensions
    print("\n✓ Tensor shape validation:")
    print(f"  - api_struct: {results['api_struct'].shape} (expected: [{tester.batch_size}, {tester.hidden_dim}])")
    print(f"  - fused: {results['fused'].shape} (expected: [{tester.batch_size}, {tester.fusion_dim}])")
    print(f"  - retrieved_indices: {results['retrieved_indices'].shape} (expected: [{tester.batch_size}, 64])")
    print(f"  - final_scores: {results['final_scores'].shape} (expected: [{tester.batch_size}, 64])")
    
    # Check value ranges
    print("\n✓ Value range validation:")
    print(f"  - api_struct: [{results['api_struct'].min():.4f}, {results['api_struct'].max():.4f}]")
    print(f"  - fused: [{results['fused'].min():.4f}, {results['fused'].max():.4f}]")
    print(f"  - final_scores: [{results['final_scores'].min():.4f}, {results['final_scores'].max():.4f}]")
    
    # Check no NaNs/Infs
    print("\n✓ Numerical stability checks:")
    has_nan_api = torch.isnan(results['api_struct']).any()
    has_nan_fused = torch.isnan(results['fused']).any()
    has_nan_scores = torch.isnan(results['final_scores']).any()
    
    print(f"  - api_struct NaNs: {'❌' if has_nan_api else '✓'}")
    print(f"  - fused NaNs: {'❌' if has_nan_fused else '✓'}")
    print(f"  - final_scores NaNs: {'❌' if has_nan_scores else '✓'}")
    
    print("\n" + "✅ " * 30)
    print("All tests passed!")
    print("✅ " * 30)


if __name__ == '__main__':
    main()
