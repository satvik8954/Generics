# ExciPick — Excipient Prediction for Generic Drug Formulations

ExciPick is a neural network that predicts which **excipients** (inactive ingredients) should be used in a generic drug formulation based on the active ingredient, dose, route, and dosage form.

## Architecture

```
API (SMILES → 20 molecular descriptors) ──→ APIEncoder (MLP)  ──┐
Dose (log-normalized mg) + Per-Unit ──→ StrengthEncoder ──────┤
                                                                ├──→ ContextFusion ──→ Scorer ──→ Excipient Scores
Route (embedding) ──────────────────────────────────────────────┤
Dosage Form (embedding) ────────────────────────────────────────┘
```

| Component | Description |
|---|---|
| `APIEncoder` | 2-layer MLP (20 → 256 → 128) on molecular descriptors |
| `StrengthEncoder` | Encodes dose + per-unit embedding → 64-dim |
| `ContextFusion` | Concatenates all encodings + route/form embeddings → 192-dim |
| `Scorer` | Scores every excipient in vocab against the context vector |

## Dataset

- **Source**: FDA drug labels (SPL/NDC), preprocessed into `Data/f3.csv`
- **Size**: ~29,000 formulations, 1,056 unique APIs, 1,299 excipients
- **Split**: 80/10/10 train/val/test by API chemical similarity clusters (Tanimoto on Morgan fingerprints) to prevent data leakage

## Project Structure

```
Generics/
├── Data/
│   ├── f3.csv                  # Raw dataset (FDA drug labels)
│   └── api_features.csv        # Precomputed molecular descriptors (generated)
├── model/
│   ├── FULL_MODEL.py           # ExciPickModel (assembles all components)
│   ├── api_encoder.py          # API molecular feature encoder
│   ├── strength_encoder.py     # Dose + per-unit encoder
│   ├── context_fusion.py       # Fuses all inputs with route/form embeddings
│   └── excipient_scorer.py     # Scores each excipient against context
├── compute_features.py         # One-time: SMILES → 20 RDKit descriptors → CSV
├── preprocess.py               # Parses data, normalizes, builds vocabs → pickle
├── split.py                    # Cluster-aware train/val/test split
├── dataset.py                  # PyTorch Dataset class
├── training.py                 # Training loop with validation
├── test.py                     # Evaluation on test set
├── metrics.py                  # Precision@K, Recall@K, F1@K, Jaccard@K
└── config.py                   # All hyperparameters
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
