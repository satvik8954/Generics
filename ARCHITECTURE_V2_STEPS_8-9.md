# ExciPick v2 Architecture: Stages 8-9 (Search & Bundle Scoring)

## Overview

This document details **Stage 8 (Search)** and **Stage 9 (Bundle Scoring)** of the ExciPick v2 architecture pipeline.

```
Stage 1-5: Retrieval Phase
   ↓
   Top-K Excipients (50-100)
   ↓
[STAGE 8: BEAM SEARCH] ← YOU ARE HERE
   ↓
   Promising Combinations (5-10)
   ↓
[STAGE 9: SET TRANSFORMER]
   ↓
   Ranked Formulations + Bundle Scores
```

---

## STAGE 8: BEAM SEARCH (Combination Generation)

### Purpose
Transform top-K individual excipients into promising **combinations** (sets) that work well together synergistically.

### Key Insight
**Excipient compatibility is NOT additive.** Two compatible individual excipients may interact poorly together, while an individually mediocre excipient may enable unexpected synergies. Beam Search finds these synergistic combinations.

### Algorithm

**Input**: 
- `top_k_indices`: (K,) indices of best excipients from retrieval
- `top_k_scores`: (K,) their retrieval scores
- `all_excipient_scores`: (num_excipients,) scores for all excipients

**Process**:
```
1. Initialize: Start with top-1 excipient (highest retrieval score)
   Beams = [{combo: [exc_0], score: 0.95}]

2. Expand: For step=1 to max_steps:
   For each active beam:
     - Try adding each unused excipient from top-K
     - Score new combination (individual avg + interaction bonus)
     - Create new candidate
   
   Sort candidates by score
   Keep top-B candidates (beam width = 5)

3. Output: Top-N combinations with their scores
```

**Key Components**:

1. **ExcipientCombinationScorer**
   - Learns pairwise interaction networks
   - Scores = (individual_avg + interaction_bonus)
   - Interaction network: 2×emb_dim → hidden → ... → 1 (scalar)

2. **BeamSearchCombinationGenerator**
   - Maintains B beams (default B=5)
   - Max combination size (default 5 excipients)
   - Supports both learned and naive scoring

3. **SearchStage2** (PyTorch Module)
   - Complete wrapper for integration
   - Handles batching
   - Returns: combinations + scores

### Hyperparameters

```python
# Key settings
num_beams = 5              # Beam width (more = more search, slower)
max_combination_size = 5   # Max excipients per combination
use_learned_scorer = True  # Use learned pairwise interactions
hidden_dim = 256           # Scorer embedding dimension
```

### Scoring Formula

```
score(combo) = sigmoid(MLP(
    [
        mean(individual_scores),
        mean(pairwise_interactions)
    ]
))
```

**Individual Score**: Average retrieval scores of excipients
**Interaction Term**: Learned neural network scoring pairwise compatibility

### Usage Example

```python
from model.v2 import SearchStage2

# Initialize
searcher = SearchStage2(
    num_beams=5,
    max_combination_size=5,
    use_learned_scorer=True,
    hidden_dim=256,
    num_excipients=1299,
    device='cuda'
)

# Forward pass
top_k_indices = torch.tensor([[10, 25, 50, 100, 200]])  # (1, 5)
top_k_scores = torch.tensor([[0.95, 0.92, 0.88, 0.85, 0.80]])  # (1, 5)
all_scores = torch.randn(1, 1299)

combinations, combination_scores = searcher(
    top_k_indices,
    top_k_scores,
    all_scores
)

# Output:
# combinations: List[List[List[int]]] = [[[10, 25], [10, 50], ...]]
# combination_scores: (batch_size, num_beams) = [[0.94, 0.91, 0.88, ...]]
```

### Trade-offs

| Beam Width | Speed | Quality | Diversity |
|---|---|---|---|
| 1 (greedy) | Fast ✓ | Lower | Low |
| 3-5 (current) | Balanced | Good ✓ | Medium |
| 10+ | Slow | Better | High |

---

## STAGE 9: SET TRANSFORMER (Bundle Scoring)

### Purpose
Score a **set** of excipients to predict overall formulation compatibility (0 to 1).

### Key Insight
Excipients form an **unordered set** with no inherent ordering. Standard RNNs/Transformers assume order. **Set Transformers** are permutation-invariant: same set → same output regardless of input order.

### Architecture

**Pipeline**:
```
Excipient Indices [10, 25, 50]
        ↓ (Embed)
  Excipient Embeddings (3×256)
        ↓ (Set Attention × 2 layers)
  Learned Pairwise Interactions
        ↓ (Aggregate: Mean/Max)
  Single Set Vector (256,)
        ↓ (MLP × 3 layers)
  Bundle Score ∈ [0, 1]
```

### Components

#### 1. **SetAttention** (Permutation-Invariant Attention)
- Multi-head self-attention adapted for sets
- Each element attends to all others
- Output: Element-wise refined representations
- Maintains permutation invariance through symmetric aggregation

```python
SetAttention(
    Q = Linear(x)     # Query
    K = Linear(x)     # Key
    V = Linear(x)     # Value (same input)
    scores = softmax(Q @ K^T)
    output = scores @ V + residual
)
```

#### 2. **SetAttentionBlock**
- Complete block: Attention + Feed-Forward
- Layer normalization + residual connections
- Can stack multiple blocks for deeper interaction modeling

#### 3. **SetTransformer**
- Stack of K attention blocks (default K=2)
- Learns hierarchical set interactions
- Aggregation methods:
  - `mean`: Average all elements (simple)
  - `max`: Maximum pooling (robust to outliers)
  - `mean+max`: Concatenate both (richer)

#### 4. **SetTransformerBundleModel**
- Integrates everything: Embed → Transform → Score
- Learnable excipient embeddings (1299 × 256)
- Projects embeddings if needed
- Final MLP: hidden → 128 → 64 → 1 → sigmoid

### Mathematical Foundation

**Permutation Invariance**:
```
∀ permutation π:
  f(x₁, x₂, ..., xₙ) = f(x_π(1), x_π(2), ..., x_π(n))
```

Set Transformers achieve this through symmetric operations (mean, max) in aggregation.

### Usage Example

```python
from model.v2 import SetTransformerBundleModel

# Initialize model
model = SetTransformerBundleModel(
    num_excipients=1299,
    excipient_emb_dim=256,
    hidden_dim=256,
    num_transformer_layers=2,
    num_heads=8,
    ff_dim=512,
    dropout=0.1
)

# Score single combination
combo_indices = [10, 25, 50]  # 3 excipients
score_single = model(combo_indices, batch=False)
# Output: scalar tensor ≈ 0.87

# Score batch of combinations
batch_combos = [
    [10, 25, 50],
    [10, 50, 100],
    [25, 100, 200]
]
# Pad to max length
padded = torch.tensor([[10, 25, 50], [10, 50, 100, 0, 0], ...])
scores_batch = model(padded, batch=True)
# Output: (3,) tensor [0.87, 0.82, 0.79]

# Alternative: score_combinations() handles padding automatically
scores = model.score_combinations(batch_combos)
```

### Hyperparameters

```python
# Architecture
excipient_emb_dim = 256            # Individual excipient embedding
hidden_dim = 256                   # Set Transformer hidden size
num_transformer_layers = 2         # Number of attention blocks
num_heads = 8                      # Attention heads per block
ff_dim = 512                       # Feed-forward hidden dim
dropout = 0.1                      # Dropout rate
bundle_hidden_dim = 128            # Final MLP hidden size
aggregation = 'mean'               # 'mean', 'max', or 'mean+max'
```

### Training Signals

**What to supervise for Stage 9**:
- Formulation compatibility (pairwise: good/bad combinations)
- Bundle success rate (% formulations accepted in preclinical testing)
- Manufacturability score (if available from domain experts)

**Loss Options**:
- Binary cross-entropy (compatible vs incompatible)
- MSE (continuous score prediction)
- Ranking loss (contrastive: good combos closer than bad)

---

## INTEGRATION: STAGE 8 → 9

### Complete Pipeline

```python
from model.v2 import SearchStage2, BundleModelStage9

# Stage 8: Generate combinations
searcher = SearchStage2(
    num_beams=5,
    max_combination_size=5
)

combinations, combination_scores = searcher(
    top_k_indices,
    top_k_scores,
    all_excipient_scores
)
# combinations: List[List[List[int]]]
# combination_scores: (batch_size, 5)

# Stage 9: Score combinations
bundle_scorer = BundleModelStage9(
    num_excipients=1299,
    hidden_dim=256,
    num_transformer_layers=2
)

# Process combinations from Stage 8
bundle_scores, ranked_formulations = bundle_scorer(
    combinations[0]  # First batch's combinations
)

# Output:
# bundle_scores: (5,) - scores for 5 combinations
# ranked_formulations: [(combo, score), ...] sorted by score DESC
```

### Data Flow

```
Batch Input
├─ API: SMILES + metadata
├─ Excipients: IDs (partial)
└─ Reference formulation: Excipient set + label

Stages 1-5: Retrieval
├─ MPNN: API → graph embedding
├─ HGT: Graph context
├─ Fusion: API + context
├─ FAISS: top-K excipients
└─ Interaction: individual scores

Stage 8: Search
├─ Beam Search: top-K → combinations
└─ Output: 5 promising combinations

Stage 9: Bundle Scoring
├─ Set Transformer: combination → embedding
├─ MLP: embedding → compatibility score
└─ Output: Ranked formulations
```

---

## FUTURE: GFlowNet Integration

Currently, Stage 8 uses **fixed beam search**. Once the model stabilizes, we can integrate **GFlowNet** (Generative Flow Networks):

### Why GFlowNet?

1. **Learned Exploration**: Learns to explore combination space efficiently instead of fixed beam width
2. **Multi-Modal Solutions**: Naturally handles multiple good solutions of different types
3. **Scalability**: Better for larger combination sizes (>10)
4. **Flow Matching**: Theoretical guarantees on exploration diversity

### Implementation Plan

```python
# Placeholder location
from model.v2 import GFlowNetSearchPlaceholder

# To implement later:
class GFlowNetCombinationGenerator(nn.Module):
    """
    GFlowNet-based search when model stabilizes.
    
    Features:
    - Learn state transitions: curr_combo → next_combo
    - Reward: combination score from Stage 9
    - Objective: Maximize total flow (trajectory reward)
    """
```

---

## Performance Considerations

### Stage 8 (Beam Search)
- **Time Complexity**: O(B × K × S) where B=beams, K=top-K, S=max_size
- **Default**: 5 beams × 50 top-K × 5 size ≈ 1,250 candidates evaluated
- **Speed**: ~10-50ms on GPU per batch

### Stage 9 (Set Transformer)
- **Time Complexity**: O(n² × heads × layers) where n=set_size
- **Default**: (5 size)² × 8 heads × 2 layers = 400 operations
- **Speed**: ~1-5ms per combination on GPU

### Total E2E
- **Stages 1-7** (Retrieval): ~50-100ms
- **Stage 8** (Search): ~25ms
- **Stage 9** (Scoring): ~5ms
- **Total**: ~80-130ms per batch of 32

---

## Files

- `beam_search.py` - Stage 8 implementation
- `set_transformer_bundle.py` - Stage 9 implementation
- `test_v2_pipeline.py` - Integration tests (to be updated)

## Next Steps

1. **Immediate**: Test both stages with synthetic data
2. **Phase 1**: Train Stage 9 on known formulation labels
3. **Phase 2**: Fine-tune Stage 8 interaction scorer with Stage 9 rewards
4. **Phase 3**: Integrate with full pipeline (Stages 1-7)
5. **Phase 4**: GFlowNet exploration once model converges
