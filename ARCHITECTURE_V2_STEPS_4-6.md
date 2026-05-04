# ExciPick v2 Architecture: Issues Filtered & Steps 4-6 Built

## ISSUES FILTERED OUT (Steps 1-3)

### Issue 1: MPNN Returns Only Graph-Level Embeddings
**Problem**: Original `mpnn.py` used only `global_add_pool`, losing node-level information needed for heterogeneous graph construction.

**Fix**: 
```python
# Now returns both:
- graph_emb: (B, hidden_dim) aggregated representation
- node_embs: (num_nodes, hidden_dim) for downstream fusion
```
- Added feature normalization: atomic numbers scaled to [0, 1]
- Combined add + mean pooling for richer graph representation
- Added residual connections with layer normalization

### Issue 2: SMILES→Graph Lacked Feature Normalization
**Problem**: Raw atomic numbers (1-118) and features had inconsistent scales, causing training instability.

**Fix**:
- Normalize atomic number: `/118.0`
- Normalize degree: `/4.0`
- Normalize formal charge: `(-2 to 2) → [0, 1]`
- Normalize hybridization: `/3.0`
- Added hydrogen atoms for complete structural information
- Used `float32` for better numerical stability

### Issue 3: HGT Incomplete & Missing Metadata Handling
**Problem**: Original `hgt.py` was skeletal—no clear node-type embeddings, no residual connections, unclear heterogeneous graph structure.

**Fix**:
- Added learnable node-type embeddings for better API vs Excipient differentiation
- Proper residual connections: `norm(x + x_new)`
- Enhanced documentation with metadata tuple structure
- Built `DualStreamHGT` as alternative for separate knowledge/similarity streams
- Clear separation: API-EXC edges vs EXC-EXC similarity edges

### Issue 4: Missing Knowledge Graph Construction
**Problem**: No integration between individual molecular graphs and the heterogeneous knowledge graph.

**Status**: Addressed by HGT improvements; knowledge graph building handled externally in dataset pipeline.

---

## STEPS 4-6: NEW MODULES BUILT

### **STEP 4: FUSION (GATED FUSION)**
**File**: `fusion_gating.py`

#### Purpose
Combines API structural encoding (MPNN) with GNN-enriched context via learned gating.

#### Key Components

**GatedFusion** (Primary)
```
g = sigmoid(W_g * [api_struct, context] + b_g)
fused = g * api_struct + (1-g) * context
```
- Projects both inputs to same dimension
- Learns adaptive weights for each modality
- Includes layer normalization and dropout

**DualModalityFusion** (Alternative)
- Multi-head fusion: separate gate per attention head
- More expressive; learns multiple fusion patterns
- Better for capturing complex interactions

#### Inputs/Outputs
```
Input:
  - api_struct: (B, struct_dim) from MPNN
  - context: (B, context_dim) from HGT + metadata

Output:
  - fused: (B, fusion_dim) combined representation
```

---

### **STEP 5: RETRIEVAL (FAISS)**
**File**: `retrieval_faiss.py`

#### Purpose
Efficient dense retrieval of candidate excipients using FAISS indexing.

#### Key Components

**FAISSRetriever** (Primary)
- Builds L2 distance FAISS index (offline)
- Per-sample: retrieve top-K nearest neighbors
- GPU support for large vocabularies

**HybridRetriever** (Alternative)
- Combines dense (FAISS) + sparse retrieval
- Better coverage for rare/niche excipients
- Rule-based filtering support

**ContextAwareRetriever** (Advanced)
- Two-stage: coarse retrieval → learned re-ranking
- Re-ranking uses context-excipient interaction features
- Balances recall and precision

#### Workflow
```
1. Build index from excipient embeddings (once, offline)
2. For each context vector, search top-K nearest neighbors
3. Return indices + similarity scores
```

#### Efficiency Gains
- Reduces O(B×V) scoring to O(B×K) where K << V
- Typical: V=1,299 excipients, retrieve K=256

---

### **STEP 6: INTERACTION MODELING**
**File**: `interaction_modeling.py`

#### Purpose
Compute rich context-excipient interactions for scoring retrieved candidates.

#### Key Components

**BilinearInteraction**
```
Score = context^T * W * excipient + b
```
- Classic bilinear interaction
- Efficient for batch scoring
- Good baseline

**MultiHeadInteraction**
- Separate interaction per attention head
- Each head learns different interaction pattern (solubility, hydrophobicity, etc.)
- Ensemble combines head scores with learned weights

**GatedInteraction**
```
Gate = sigmoid(W_g * [context, exc])
Score = Gate * Bilinear(context, exc)
```
- Learned gating determines relevance
- More selective scoring mechanism

**InteractionHead** (Ensemble)
- Combines all three interaction types
- Learned softmax ensemble for weighted combination
- Most expressive; best for complex patterns

**InteractionModule** (Complete Pipeline)
```
Input:
  - context: (B, fusion_dim) from step 4
  - retrieved_indices: (B, K) from step 5
  - exc_embeddings: (V, exc_dim) from HGT

Output:
  - scores: (B, K) refined interaction scores
```

#### Modes
```
'weighted': Full ensemble (bilinear + multihead + gated)
'multihead': Multi-head attention interactions
'gated': Learned gating mechanism
'bilinear': Simple bilinear baseline
```

---

## ARCHITECTURE SUMMARY: INPUT → OUTPUT

```
1. INPUT (SMILES)
   ↓
2. STRUCTURAL ENCODING (MPNN)
   graph_emb: (B, 256) + node_embs: (nodes, 256)
   ↓
3. GRAPH ENCODING (HGT)
   Enriches via heterogeneous knowledge graph
   → enriched_api: (num_apis, 256)
   → enriched_exc: (V, 256)
   ↓
4. FUSION (Gated Fusion)
   Combines structural + graph signals
   fused_context: (B, 256)
   ↓
5. RETRIEVAL (FAISS)
   Dense search: top-K candidates
   candidate_indices: (B, 256)
   ↓
6. INTERACTION MODELING
   Score candidates given context
   final_scores: (B, 256)
   ↓
7. OUTPUT (next: scoring & ranking)
```

---

## NEXT STEPS (Steps 7-11)

Once you're satisfied with 1-6, build:
- **Step 7**: SCORING (SPAK SCORER) - Final ranking with diversity penalties
- **Step 8**: STAGE 2 SEARCH - Secondary refinement with combo generation
- **Step 9**: BUNDLE MODEL - Multi-excipient bundle selection
- **Step 10**: API ENGINE - Rule-based filtering and soft penalties
- **Step 11**: OUTPUT - Final formulation recommendations

---

## INTEGRATION NOTES

### Model Parameters (Update `config.py`)
```python
# Step 4: Fusion
"fusion_dim": 256,
"fusion_heads": 4,  # For DualModalityFusion

# Step 5: Retrieval
"retrieval_top_k": 256,
"faiss_gpu": True,

# Step 6: Interaction
"interaction_dim": 256,
"interaction_heads": 8,
"interaction_mode": "weighted",  # or "gated", "multihead", "bilinear"
```

### Data Pipeline
```python
# 1. Build knowledge graph with API-EXC and EXC-EXC edges
heterograph = build_heterograph(apis, excipients, knowledge_base)

# 2. Run through steps 1-3
mpnn_model = StructMPNN()
hgt_model = HeteroGraphTransformer(metadata=heterograph.metadata)
api_embs, exc_embs = run_gnn_pipeline(heterograph, mpnn_model, hgt_model)

# 3. At inference time
for batch in dataloader:
    # Step 4: Fusion
    fused = fusion_model(api_embs[batch], context)
    
    # Step 5: Retrieval
    retrieval_model.build_index(exc_embs)
    indices, sim_scores = retrieval_model(fused, top_k=256)
    
    # Step 6: Interaction
    final_scores = interaction_model(fused, indices, exc_embs)
```

---

## Files Modified

- ✅ `model/v2/smiles_graph.py` - Feature normalization, hydrogen addition
- ✅ `model/v2/mpnn.py` - Graph + node embeddings, dual pooling
- ✅ `model/v2/hgt.py` - Node-type embeddings, residual connections, DualStreamHGT
- ✅ `model/v2/fusion_gating.py` - NEW: GatedFusion, DualModalityFusion
- ✅ `model/v2/retrieval_faiss.py` - NEW: FAISSRetriever, HybridRetriever, ContextAwareRetriever
- ✅ `model/v2/interaction_modeling.py` - NEW: 5 interaction classes + complete module
- ✅ `model/v2/__init__.py` - Updated with all new exports
