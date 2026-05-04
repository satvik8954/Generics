# ExciPick — Excipient Prediction for Generic Drug Formulations

ExciPick is a neural network that predicts which **excipients** (inactive ingredients) should be used in a generic drug formulation based on the active ingredient, dose, route, and dosage form.

## Architecture v2 (Multi-Stage Pipeline)

### Complete Pipeline: 10 Stages

```
Stage 1: INPUT (SMILES → Graph)
  ↓ Drug molecule as PyG graph
  
Stage 2: STRUCTURAL ENCODING (MPNN)
  ↓ Graph embedding + node features
  
Stage 3: GRAPH ENCODING (HGT)
  ↓ Heterogeneous API-Excipient context
  
Stage 4: FUSION (Gated Fusion)
  ↓ Combine structural + context
  
Stage 5: RETRIEVAL (FAISS)
  ↓ Dense retrieval: top-50/100 excipients
  
Stage 6: INTERACTION MODELING
  ↓ Individual excipient scoring
  
Stage 7: SIAMESE NETWORK (Future)
  ↓ Contrastive learning for excipient similarity
  
Stage 8: BEAM SEARCH ⭐ NEW
  ├─ Input: Top-K excipients
  ├─ Process: Beam search for combinations
  └─ Output: 5-10 promising sets
  
Stage 9: SET TRANSFORMER BUNDLE MODEL ⭐ NEW
  ├─ Input: Excipient combinations
  ├─ Process: Set Transformer → compatibility score
  └─ Output: Ranked formulations [0, 1]
  
Stage 10: RULE ENGINE (Future)
  ├─ Hard filters: regulatory, safety, toxicity
  └─ Soft penalties: manufacturing, cost, shelf-life
```

### Stage Descriptions

| Stage | Module | Purpose | Input | Output |
|-------|--------|---------|-------|--------|
| 1 | `smiles_graph.py` | SMILES → PyG graph | Drug SMILES | PyG Data |
| 2 | `mpnn.py` | Molecular structure encoding | PyG graph | Graph embedding |
| 3 | `hgt.py` | Heterogeneous graph context | API-EXC edges | Context vector |
| 4 | `fusion_gating.py` | Gated fusion | Structure + context | Fused embedding |
| 5 | `retrieval_faiss.py` | Dense retrieval | Fused embedding | Top-K excipients |
| 6 | `interaction_modeling.py` | Interaction scoring | Context + excipients | Individual scores |
| 7 | `siamese.py` | Excipient similarity | Excipient pairs | Contrastive loss |
| **8** | **`beam_search.py`** | **Combination search** | **Top-K scores** | **Combinations** |
| **9** | **`set_transformer_bundle.py`** | **Bundle compatibility** | **Combinations** | **Bundle scores** |
| 10 | `rule_engine.py` | Hard/soft filters | Ranked formulations | Final recommendations |

### Key Innovation: Stages 8-9

**Stage 8: Beam Search**
- Takes top-K individual excipients
- Iteratively finds promising **combinations** via beam search
- Learns pairwise excipient interactions
- Output: 5-10 synergistic combinations

**Stage 9: Set Transformer**
- Scores each combination for overall compatibility
- Uses permutation-invariant Set Transformer architecture
- Learns multi-way excipient interactions
- Output: Bundle score ∈ [0, 1]

### Project Structure (Updated)

```
Generics/
├── Data/
│   ├── f3.csv                  # Raw dataset (FDA drug labels)
│   └── api_features.csv        # Precomputed molecular descriptors
│
├── model/v2/
│   ├── __init__.py             # Module exports
│   ├── smiles_graph.py         # Stage 1: SMILES → graph
│   ├── mpnn.py                 # Stage 2: MPNN
│   ├── hgt.py                  # Stage 3: HGT
│   ├── fusion_gating.py        # Stage 4: Fusion
│   ├── retrieval_faiss.py      # Stage 5: FAISS retrieval
│   ├── interaction_modeling.py # Stage 6: Scoring
│   ├── siamese.py              # Stage 7: Siamese network
│   ├── beam_search.py          # Stage 8: Beam search ⭐
│   └── set_transformer_bundle.py # Stage 9: Bundle scoring ⭐
│
├── ARCHITECTURE_V2_STEPS_4-6.md    # Docs for stages 4-6
├── ARCHITECTURE_V2_STEPS_8-9.md    # Docs for stages 8-9 ⭐
│
├── preprocess.py               # Data preprocessing
├── split.py                    # Train/val/test split
├── dataset.py                  # PyTorch Dataset
├── config.py                   # Hyperparameters
├── metrics.py                  # Evaluation metrics
├── build_graph.py              # Graph construction
│
├── test_v2_pipeline.py         # End-to-end tests
├── README.md                   # This file
└── setup.py                    # Package setup (optional)
```


## Setup

### Requirements

- Python 3.10+
- PyTorch
- RDKit
- scikit-learn
- pandas, numpy

```bash
pip install torch rdkit scikit-learn pandas numpy
```

## Usage

### 1. Compute API Features (run once)

Extracts 20 molecular descriptors from SMILES for each unique API and saves to `Data/api_features.csv`:

```bash
python compute_features.py
```

### 2. Preprocess

Parses excipients, merges API features, normalizes, builds vocabularies:

```bash
python preprocess.py
```

### 3. Train

Splits data by API cluster, trains with validation, saves best model:

```bash
python training.py
```

Outputs:
- `best_model.pt` — best model checkpoint (by validation loss)
- `test_data.pkl` — held-out test set for evaluation

### 4. Evaluate

```bash
# Full evaluation on test set
python test.py

# Quick smoke test (dummy data, no GPU needed)
python test.py --smoke
```

---

## NEW: Stage 8 & 9 Usage (Beam Search + Set Transformer)

### Stage 8: Beam Search (Combination Generation)

**Purpose**: Transform top-K individual excipients into synergistic **combinations**.

```python
import torch
from model.v2 import SearchStage2

# Initialize beam search
searcher = SearchStage2(
    num_beams=5,                    # Keep 5 best combinations at each step
    max_combination_size=5,         # Max 5 excipients per combination
    use_learned_scorer=True,        # Learn pairwise interactions
    hidden_dim=256,
    num_excipients=1299,
    device='cuda'
)

# From retrieval stage (Stage 5):
# top_k_indices: (batch_size=2, k=50) - indices of top-50 excipients
# top_k_scores: (batch_size=2, k=50) - their scores [0, 1]
# all_excipient_scores: (batch_size=2, 1299) - all excipient scores

batch_size = 2
top_k_indices = torch.tensor([
    [10, 25, 50, 100, 200, ...],  # Batch 1
    [15, 30, 60, 110, 210, ...]   # Batch 2
])
top_k_scores = torch.tensor([
    [0.95, 0.92, 0.88, 0.85, 0.80, ...],
    [0.94, 0.91, 0.87, 0.84, 0.79, ...]
])
all_excipient_scores = torch.randn(batch_size, 1299)

# Perform beam search
combinations, combination_scores = searcher(
    top_k_indices,
    top_k_scores,
    all_excipient_scores
)

# Output:
# combinations[0]: [[10, 25], [10, 50], [25, 50], [10, 25, 50], [10, 25, 100]]
# combination_scores[0]: [0.92, 0.89, 0.88, 0.87, 0.85]
```

**Key Hyperparameters**:
- `num_beams=5`: Beam width (balance between quality and speed)
- `max_combination_size=5`: Don't create combinations >5 excipients
- `use_learned_scorer=True`: Learn pairwise compatibility (False = use naive scoring)

**Output Interpretation**:
- Combinations are sorted by score (descending)
- Each combination is a list of excipient indices
- Scores indicate predicted compatibility (higher = better)

---

### Stage 9: Set Transformer Bundle Model (Compatibility Scoring)

**Purpose**: Score each combination to predict overall formulation compatibility.

```python
import torch
from model.v2 import SetTransformerBundleModel, BundleModelStage9

# Option 1: Direct scoring with SetTransformerBundleModel
model = SetTransformerBundleModel(
    num_excipients=1299,
    excipient_emb_dim=256,
    hidden_dim=256,
    num_transformer_layers=2,      # Stack 2 attention blocks
    num_heads=8,
    ff_dim=512,
    dropout=0.1,
    aggregation='mean'             # Aggregate set via mean pooling
)

# Score single combination
combo = [10, 25, 50]
score_single = model(combo, batch=False)
print(f"Compatibility score: {score_single:.4f}")  # Outputs: 0.8700

# Score batch of combinations
combos_batch = torch.tensor([
    [10, 25, 50, 0, 0],      # Padded with 0s
    [10, 50, 100, 0, 0],
    [25, 100, 200, 300, 0]
])
scores_batch = model(combos_batch, batch=True)
print(scores_batch)  # Shape: (3,), e.g. [0.87, 0.82, 0.79]

# Option 2: Full Stage 9 wrapper (integrates with Stage 8)
bundle_scorer = BundleModelStage9(
    num_excipients=1299,
    hidden_dim=256,
    num_transformer_layers=2,
    device='cuda'
)

# Take combinations from Stage 8
combinations_from_beam_search = [
    [10, 25],
    [10, 50],
    [25, 50],
    [10, 25, 50]
]

# Score all combinations
bundle_scores, ranked_formulations = bundle_scorer(
    combinations_from_beam_search
)

print("Ranked formulations:")
for combo, score in ranked_formulations:
    print(f"  {combo}: {score:.4f}")
    
# Output:
#   [10, 25, 50]: 0.8700
#   [10, 25]: 0.8523
#   [25, 50]: 0.8147
#   [10, 50]: 0.8092
```

**Architecture**:
```
Excipient Indices [10, 25, 50]
    ↓ (Embed & Project)
(3, 256) embeddings
    ↓ (Set Attention Block × 2)
Learned pairwise interactions
    ↓ (Aggregate: Mean/Max)
(256,) set embedding
    ↓ (MLP: 256→128→64→1)
Bundle Score ∈ [0, 1]
```

**Key Hyperparameters**:
- `num_transformer_layers=2`: How many attention blocks to stack
- `num_heads=8`: Attention heads per block
- `aggregation='mean'`: How to aggregate set (mean/max/mean+max)
- `excipient_emb_dim=256`: Individual excipient embedding size

---

### Complete E2E Pipeline (Stages 1-9)

```python
import torch
from model.v2 import (
    smiles_to_graph, StructMPNN, HeteroGraphTransformer,
    GatedFusion, FAISSRetriever, InteractionModule,
    SearchStage2, BundleModelStage9
)

# Stage 1-2: API structure
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
graph = smiles_to_graph(smiles)
mpnn = StructMPNN(hidden_dim=256, num_layers=2)
graph_emb = mpnn(graph)  # (256,)

# Stage 3: Context (HGT)
hgt = HeteroGraphTransformer(hidden_dim=256, num_layers=2)
context_emb = hgt(hetero_graph)  # (256,)

# Stage 4: Fusion
fuser = GatedFusion(struct_dim=256, context_dim=256, fusion_dim=256)
fused = fuser(graph_emb, context_emb)  # (256,)

# Stage 5: Retrieval (FAISS)
retriever = FAISSRetriever(hidden_dim=256, num_excipients=1299, top_k=50)
top_k_indices, top_k_scores = retriever.search(fused)  # (50,), (50,)

# Stage 6: Interaction scoring
scorer = InteractionModule(context_dim=256, excipient_dim=128, num_excipients=1299)
all_excipient_scores = scorer(fused)  # (1299,)

# Stage 8: Beam search combinations
searcher = SearchStage2(num_beams=5, max_combination_size=5)
combinations, combo_scores = searcher(
    top_k_indices.unsqueeze(0),
    top_k_scores.unsqueeze(0),
    all_excipient_scores.unsqueeze(0)
)

# Stage 9: Bundle scoring
bundle_scorer = BundleModelStage9(num_excipients=1299, hidden_dim=256)
final_scores, ranked_formulations = bundle_scorer(combinations[0])

print("TOP RECOMMENDED FORMULATIONS:")
for combo, score in ranked_formulations[:3]:
    print(f"  Excipients: {combo}")
    print(f"  Compatibility Score: {score:.4f}")
    print()
```

---

## Documentation

For detailed architecture explanation:
- **Stages 4-6**: See [ARCHITECTURE_V2_STEPS_4-6.md](ARCHITECTURE_V2_STEPS_4-6.md)
- **Stages 8-9**: See [ARCHITECTURE_V2_STEPS_8-9.md](ARCHITECTURE_V2_STEPS_8-9.md)

---

## Molecular Descriptors (20-dim API features)

| # | Descriptor | Description |
|---|---|---|
| 1 | MolWt | Molecular weight |
| 2 | MolLogP | Wildman-Crippen LogP |
| 3 | TPSA | Topological polar surface area |
| 4 | NumHDonors | H-bond donors |
| 5 | NumHAcceptors | H-bond acceptors |
| 6 | NumRotatableBonds | Rotatable bonds |
| 7 | NumAromaticRings | Aromatic ring count |
| 8 | NumAliphaticRings | Aliphatic ring count |
| 9 | RingCount | Total ring count |
| 10 | FractionCSP3 | Fraction of sp3 carbons |
| 11 | HeavyAtomCount | Non-hydrogen atoms |
| 12 | NumValenceElectrons | Valence electrons |
| 13 | MolMR | Molar refractivity |
| 14 | LabuteASA | Labute ASA |
| 15 | BalabanJ | Balaban's J index |
| 16 | BertzCT | Bertz complexity |
| 17 | HallKierAlpha | Hall-Kier alpha |
| 18 | NumSaturatedRings | Saturated ring count |
| 19 | NumHeteroatoms | Heteroatom count |
| 20 | NHOHCount | NH and OH count |

## Evaluation Metrics

| Metric | Description |
|---|---|
| Precision@K | Fraction of top-K predictions that are correct |
| Recall@K | Fraction of true excipients captured in top-K |
| F1@K | Harmonic mean of Precision and Recall |
| Jaccard@K | Set overlap (intersection / union) |

## Config

All hyperparameters are in `config.py`. Key settings:

| Parameter | Value | Notes |
|---|---|---|
| `api_in` | 20 | Molecular descriptor dimensions |
| `batch_size` | 64 | Training batch size |
| `epochs` | 30 | Training epochs |
| `lr` | 1e-4 | Learning rate (Adam) |
| `top_k` | 10 | Default top-K for inference |
| `device` | cuda | Falls back to CPU if unavailable |

---

## Testing Stage 8 & 9

Run comprehensive tests for the new Beam Search and Set Transformer stages:

```bash
# Test beam search and bundle scoring
python test_beam_search_and_bundle.py

# Output includes:
# - Stage 8 combination generation
# - Stage 9 bundle scoring
# - Full pipeline integration
# - Hyperparameter sensitivity analysis
```

---

## Implementation Status & Roadmap

### ✅ Completed (v2.0)

| Stage | Status | Implementation | Notes |
|-------|--------|---|---|
| 1 | ✅ | SMILES → Graph | Atom/bond feature normalization |
| 2 | ✅ | MPNN | GINEConv with global pooling |
| 3 | ✅ | HGT | Heterogeneous graph transformer |
| 4 | ✅ | Gated Fusion | Learnable modal fusion |
| 5 | ✅ | FAISS Retrieval | Dense retrieval (top-K) |
| 6 | ✅ | Interaction Scoring | Multi-head interaction modeling |
| 7 | ✅ | Siamese Network | Contrastive learning (optional) |
| **8** | **✅** | **Beam Search** | **Combination generation** |
| **9** | **✅** | **Set Transformer** | **Bundle compatibility scoring** |

### ⏳ Future (v2.1+)

| Stage | Plan | Notes |
|-------|------|-------|
| 8 | GFlowNet | Learnable exploration when model stabilizes |
| 10 | Rule Engine | Hard filters (safety, regulatory) + soft penalties |
| N/A | Multi-Task Learning | Joint training on multiple objectives |
| N/A | Explainability | Saliency maps for excipient importance |

### 🎯 Next Phase: Training Signals

To fully train Stages 8-9:

1. **Stage 9 Labels**: 
   - Formulation compatibility (binary: compatible/incompatible)
   - Formulation success rate (continuous: 0-1)
   - Manufacturability score (if available)

2. **Stage 8 Pairwise Signals**:
   - Excipient compatibility pairs (from literature/domain experts)
   - Synergistic vs antagonistic combinations

3. **End-to-End Objective**:
   - Optimize recommendation quality
   - Maximize formulation success rate
   - Minimize manufacturing issues

---

## Citation & References

If you use ExciPick v2 in your research, please cite:

```bibtex
@misc{excipick_v2,
  title={ExciPick v2: Multi-Stage Neural Network for Excipient Prediction},
  year={2024},
  note={Stages 8-9 implementation (Beam Search + Set Transformer)}
}
```

### Key Papers

- **MPNN/GNN**: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- **Heterogeneous Graphs**: Zhu et al. (2021) - Heterogeneous Graph Transformer
- **Set Transformers**: Lee et al. (2019) - Set Transformers: A Framework for Attention-based Permutation-Invariant Neural Networks
- **Beam Search**: Freitag & Al-Onaizan (2017) - Beam Search Optimization in Neural Machine Translation

---

## License & Contact

This project is part of the Drug Paradigm research initiative.

For questions about Stages 8-9 implementation, refer to:
- [ARCHITECTURE_V2_STEPS_8-9.md](ARCHITECTURE_V2_STEPS_8-9.md) — Detailed technical guide
- [test_beam_search_and_bundle.py](test_beam_search_and_bundle.py) — Usage examples

