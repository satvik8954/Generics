import os
import pandas as pd
import numpy as np
import json
import pickle
from collections import Counter
INPUT_PATH = "f3.csv"
FEATURES_PATH = "api_features.csv"
OUTPUT_PATH = "processed_data.pkl"

MIN_EXCIPIENTS = 2
MAX_EXCIPIENTS = 30
EXCIPIENT_MIN_FREQ = 3

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons",
    "MolMR", "LabuteASA", "BalabanJ", "BertzCT", "HallKierAlpha",
    "NumSaturatedRings", "NumHeteroatoms", "NHOHCount",
]

def parse_excipient_names(json_str):
    """Extract excipient names from inactive_ingredients JSON string."""
    try:
        ingredients = json.loads(json_str)
        names = list(set(item["name"] for item in ingredients))
        return names
    except Exception:
        return []

def extract_smiles(json_str):
    """Extract the first SMILES string from api_smiles_map JSON."""
    try:
        smap = json.loads(json_str)
        return list(smap.values())[0]
    except Exception:
        return None


print("=" * 50)
print("ExciPick Preprocessing")
print("=" * 50)

# Check that features CSV exists
if not os.path.exists(FEATURES_PATH):
    print(f"\n[ERROR] {FEATURES_PATH} not found!")
    print("        Run `python compute_features.py` first to generate it.")
else:
    # 1. Load raw data + precomputed features
    df = pd.read_csv(INPUT_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    print(f"\n[1] Loaded: {len(df)} rows, {len(features_df)} API feature vectors")

    # 2. Parse excipients
    df["excipient_list"] = df["inactive_ingredients"].apply(parse_excipient_names)
    bad_exc = df["excipient_list"].apply(len) == 0
    print(f"[2] Rows with no parseable excipients: {bad_exc.sum()}")
    df = df[~bad_exc]

    # 3. Filter excipient count
    df = df[df["excipient_list"].apply(len) >= MIN_EXCIPIENTS]
    df = df[df["excipient_list"].apply(len) <= MAX_EXCIPIENTS]
    print(f"[3] After excipient count filter ({MIN_EXCIPIENTS}-{MAX_EXCIPIENTS}): {len(df)} rows")

    # 4. Merge precomputed API features
    df = df.merge(features_df, on="api_unii", how="left")

    # features_df has a 'smiles' column; if df already has one from raw data, merge renames it
    if "smiles_y" in df.columns:
        df["smiles"] = df["smiles_y"].fillna(df.get("smiles_x"))
        df.drop(columns=["smiles_x", "smiles_y"], inplace=True)

    missing_features = df[DESCRIPTOR_NAMES[0]].isna().sum()
    print(f"[4] Merged API features. Rows missing features: {missing_features}")
    df = df.dropna(subset=[DESCRIPTOR_NAMES[0]])

    # Pack descriptor columns into a single list column
    df["api_descriptors"] = df[DESCRIPTOR_NAMES].values.tolist()

    # 5. Normalize dose_mg (log + z-score)
    df["log_dose_mg"] = np.log10(df["dose_mg"].clip(lower=1e-9) + 1e-9)
    dose_mean = df["log_dose_mg"].mean()
    dose_std = df["log_dose_mg"].std()
    df["dose_normalized"] = (df["log_dose_mg"] - dose_mean) / dose_std
    print(f"[5] Dose normalized: mean={dose_mean:.3f}, std={dose_std:.3f}")

    # 6. Normalize API descriptors (z-score per feature)
    desc_array = np.array(df["api_descriptors"].tolist())
    desc_mean = desc_array.mean(axis=0)
    desc_std = desc_array.std(axis=0)
    desc_std[desc_std == 0] = 1.0  # avoid division by zero
    desc_normalized = (desc_array - desc_mean) / desc_std
    df["api_features"] = list(desc_normalized)
    print(f"[6] API descriptors normalized: shape={desc_array.shape}")

    # 7. Encode per_unit (denominator_unit)
    per_unit_vocab = {u: i for i, u in enumerate(sorted(df["denominator_unit"].unique()))}
    df["per_unit_id"] = df["denominator_unit"].map(per_unit_vocab)
    print(f"[7] Per-unit vocab ({len(per_unit_vocab)}): {per_unit_vocab}")

    # 8. Encode route
    route_vocab = {r: i for i, r in enumerate(sorted(df["route"].unique()))}
    df["route_id"] = df["route"].map(route_vocab)
    print(f"[8] Route vocab ({len(route_vocab)}): {list(route_vocab.keys())}")

    # 9. Encode dosage form
    form_vocab = {f: i for i, f in enumerate(sorted(df["primary_dosage_form"].unique()))}
    df["form_id"] = df["primary_dosage_form"].map(form_vocab)
    print(f"[9] Form vocab ({len(form_vocab)}): {len(form_vocab)} unique forms")

    # 10. Build excipient vocabulary (freq >= EXCIPIENT_MIN_FREQ)
    exc_counter = Counter()
    for exc_list in df["excipient_list"]:
        exc_counter.update(exc_list)

    excipient_vocab = {exc: i for i, (exc, count) in
                       enumerate(exc_counter.most_common())
                       if count >= EXCIPIENT_MIN_FREQ}

    print(f"[10] Excipient vocab: {len(excipient_vocab)} (from {len(exc_counter)} total, freq>={EXCIPIENT_MIN_FREQ})")

    # Map excipient lists to IDs
    def map_excipients(lst):
        return [excipient_vocab[e] for e in lst if e in excipient_vocab]

    df["excipient_ids"] = df["excipient_list"].apply(map_excipients)

    # Drop rows with too few mapped excipients
    df = df[df["excipient_ids"].apply(len) >= MIN_EXCIPIENTS]
    print(f"    After vocab mapping filter: {len(df)} rows")

    # 11. Rename api_unii for compatibility with split.py
    df["unii"] = df["api_unii"]

    # 12. Save
    output = {
        "df": df.reset_index(drop=True),
        "excipient_vocab": excipient_vocab,
        "route_vocab": route_vocab,
        "form_vocab": form_vocab,
        "per_unit_vocab": per_unit_vocab,
        "dose_stats": {"mean": dose_mean, "std": dose_std},
        "desc_stats": {"mean": desc_mean.tolist(), "std": desc_std.tolist()},
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)

    print(f"\n{'=' * 50}")
    print(f"[OK] Preprocessing complete!")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"   Final dataset: {len(df)} rows")
    print(f"   Excipient vocab size: {len(excipient_vocab)}")
    print(f"   Route vocab size: {len(route_vocab)}")
    print(f"   Form vocab size: {len(form_vocab)}")
    print(f"   Per-unit vocab size: {len(per_unit_vocab)}")
    print(f"{'=' * 50}")




from google.colab import drive
drive.mount('/content/drive')

CONFIG = {

    # ========================
    # MODEL SIZE (balanced)
    # ========================

    "api_in": 20,         # 20 molecular descriptors
    "api_hidden": 256,
    "api_out": 128,

    "per_unit_vocab": 4,  # {1, mL, g, L}
    "per_unit_emb": 16,
    "strength_out": 64,

    "route_vocab": 16,    # 14 routes in data, pad to 16
    "route_emb": 32,

    "form_vocab": 110,    # 103 dosage forms in data, pad to 110
    "form_emb": 32,

    "context_out": 192,

    "excipient_emb": 128,

    "scorer_hidden": 128,

    # ========================
    # GNN (HetGNN)
    # ========================

    "gnn_hidden": 128,
    "gnn_layers": 2,
    "gnn_dropout": 0.2,
    "jaccard_threshold": 0.1,
    "similarity_threshold": 0.6,

    # ========================
    # REGULARIZATION
    # ========================

    "dropout_api": 0.1,
    "dropout_context": 0.2,
    "dropout_scorer": 0.2,

    # ========================
    # TRAINING
    # ========================

    "lr": 5e-4,
    "batch_size": 64,
    "epochs": 50,

    # ========================
    # INFERENCE
    # ========================

    "top_k": 10,
    "threshold": 0.5,

    # ========================
    # SYSTEM
    # ========================

    "device": "cuda",
    "seed": 42
}

"""
preprocess.py -- Prepares Data/f3.csv for the ExciPick model.

Requires: Run `python compute_features.py` ONCE first to generate Data/api_features.csv.

Pipeline:
  1. Load raw CSV + precomputed API features
  2. Parse inactive_ingredients -> excipient names
  3. Merge API features (20 molecular descriptors)
  4. Normalize dose_mg (log + z-score) and descriptors (z-score)
  5. Encode per_unit, route, dosage_form as integer IDs
  6. Build excipient vocabulary (freq >= 3)
  7. Save processed DataFrame + vocab mappings as pickle
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
from collections import Counter

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
INPUT_PATH = "f3.csv"
FEATURES_PATH = "api_features.csv"
OUTPUT_PATH = "processed_data.pkl"

MIN_EXCIPIENTS = 2
MAX_EXCIPIENTS = 30
EXCIPIENT_MIN_FREQ = 3

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons",
    "MolMR", "LabuteASA", "BalabanJ", "BertzCT", "HallKierAlpha",
    "NumSaturatedRings", "NumHeteroatoms", "NHOHCount",
]


def parse_excipient_names(json_str):
    """Extract excipient names from inactive_ingredients JSON string."""
    try:
        ingredients = json.loads(json_str)
        names = list(set(item["name"] for item in ingredients))
        return names
    except Exception:
        return []


def extract_smiles(json_str):
    """Extract the first SMILES string from api_smiles_map JSON."""
    try:
        smap = json.loads(json_str)
        return list(smap.values())[0]
    except Exception:
        return None


# -----------------------------------------------
# MAIN PREPROCESSING
# -----------------------------------------------
def main():
    print("=" * 50)
    print("ExciPick Preprocessing")
    print("=" * 50)

    # Check that features CSV exists
    if not os.path.exists(FEATURES_PATH):
        print(f"\n[ERROR] {FEATURES_PATH} not found!")
        print("        Run `python compute_features.py` first to generate it.")
        return

    # 1. Load raw data + precomputed features
    df = pd.read_csv(INPUT_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    print(f"\n[1] Loaded: {len(df)} rows, {len(features_df)} API feature vectors")

    # 2. Parse excipients
    df["excipient_list"] = df["inactive_ingredients"].apply(parse_excipient_names)
    bad_exc = df["excipient_list"].apply(len) == 0
    print(f"[2] Rows with no parseable excipients: {bad_exc.sum()}")
    df = df[~bad_exc]

    # 3. Filter excipient count
    df = df[df["excipient_list"].apply(len) >= MIN_EXCIPIENTS]
    df = df[df["excipient_list"].apply(len) <= MAX_EXCIPIENTS]
    print(f"[3] After excipient count filter ({MIN_EXCIPIENTS}-{MAX_EXCIPIENTS}): {len(df)} rows")

    # 4. Merge precomputed API features
    df = df.merge(features_df, on="api_unii", how="left")

    # features_df has a 'smiles' column; if df already has one from raw data, merge renames it
    if "smiles_y" in df.columns:
        df["smiles"] = df["smiles_y"].fillna(df.get("smiles_x"))
        df.drop(columns=["smiles_x", "smiles_y"], inplace=True)

    missing_features = df[DESCRIPTOR_NAMES[0]].isna().sum()
    print(f"[4] Merged API features. Rows missing features: {missing_features}")
    df = df.dropna(subset=[DESCRIPTOR_NAMES[0]])

    # Pack descriptor columns into a single list column
    df["api_descriptors"] = df[DESCRIPTOR_NAMES].values.tolist()

    # 5. Normalize dose_mg (log + z-score)
    df["log_dose_mg"] = np.log10(df["dose_mg"].clip(lower=1e-9) + 1e-9)
    dose_mean = df["log_dose_mg"].mean()
    dose_std = df["log_dose_mg"].std()
    df["dose_normalized"] = (df["log_dose_mg"] - dose_mean) / dose_std
    print(f"[5] Dose normalized: mean={dose_mean:.3f}, std={dose_std:.3f}")

    # 6. Normalize API descriptors (z-score per feature)
    desc_array = np.array(df["api_descriptors"].tolist())
    desc_mean = desc_array.mean(axis=0)
    desc_std = desc_array.std(axis=0)
    desc_std[desc_std == 0] = 1.0  # avoid division by zero
    desc_normalized = (desc_array - desc_mean) / desc_std
    df["api_features"] = list(desc_normalized)
    print(f"[6] API descriptors normalized: shape={desc_array.shape}")

    # 7. Encode per_unit (denominator_unit)
    per_unit_vocab = {u: i for i, u in enumerate(sorted(df["denominator_unit"].unique()))}
    df["per_unit_id"] = df["denominator_unit"].map(per_unit_vocab)
    print(f"[7] Per-unit vocab ({len(per_unit_vocab)}): {per_unit_vocab}")

    # 8. Encode route
    route_vocab = {r: i for i, r in enumerate(sorted(df["route"].unique()))}
    df["route_id"] = df["route"].map(route_vocab)
    print(f"[8] Route vocab ({len(route_vocab)}): {list(route_vocab.keys())}")

    # 9. Encode dosage form
    form_vocab = {f: i for i, f in enumerate(sorted(df["primary_dosage_form"].unique()))}
    df["form_id"] = df["primary_dosage_form"].map(form_vocab)
    print(f"[9] Form vocab ({len(form_vocab)}): {len(form_vocab)} unique forms")

    # 10. Build excipient vocabulary (freq >= EXCIPIENT_MIN_FREQ)
    exc_counter = Counter()
    for exc_list in df["excipient_list"]:
        exc_counter.update(exc_list)

    excipient_vocab = {exc: i for i, (exc, count) in
                       enumerate(exc_counter.most_common())
                       if count >= EXCIPIENT_MIN_FREQ}

    print(f"[10] Excipient vocab: {len(excipient_vocab)} (from {len(exc_counter)} total, freq>={EXCIPIENT_MIN_FREQ})")

    # Map excipient lists to IDs
    def map_excipients(lst):
        return [excipient_vocab[e] for e in lst if e in excipient_vocab]

    df["excipient_ids"] = df["excipient_list"].apply(map_excipients)

    # Drop rows with too few mapped excipients
    df = df[df["excipient_ids"].apply(len) >= MIN_EXCIPIENTS]
    print(f"    After vocab mapping filter: {len(df)} rows")

    # 11. Rename api_unii for compatibility with split.py
    df["unii"] = df["api_unii"]

    # 12. Save
    output = {
        "df": df.reset_index(drop=True),
        "excipient_vocab": excipient_vocab,
        "route_vocab": route_vocab,
        "form_vocab": form_vocab,
        "per_unit_vocab": per_unit_vocab,
        "dose_stats": {"mean": dose_mean, "std": dose_std},
        "desc_stats": {"mean": desc_mean.tolist(), "std": desc_std.tolist()},
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)

    print(f"\n{'=' * 50}")
    print(f"[OK] Preprocessing complete!")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"   Final dataset: {len(df)} rows")
    print(f"   Excipient vocab size: {len(excipient_vocab)}")
    print(f"   Route vocab size: {len(route_vocab)}")
    print(f"   Form vocab size: {len(form_vocab)}")
    print(f"   Per-unit vocab size: {len(per_unit_vocab)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

"""
split.py — Cluster-aware train/val/test split for ExciPick.

Groups APIs by chemical similarity (Tanimoto on Morgan fingerprints)
to prevent data leakage between splits.

Split: 80% train / 10% val / 10% test (by API cluster).
"""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupShuffleSplit


def get_morgan_fp(smiles):
    """Generates 2048-bit Morgan Fingerprint (radius 2)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None


def assign_clusters(df, threshold=0.6):
    """
    Groups APIs by chemical similarity to prevent leakage.
    Uses 'unii' as the unique API identifier and 'smiles' for fingerprints.
    """
    # 1. Extract unique APIs and their SMILES
    apis = df[["unii", "smiles"]].drop_duplicates().dropna(subset=["smiles"]).reset_index(drop=True)
    apis["fp"] = apis["smiles"].apply(get_morgan_fp)

    # 2. Build adjacency list for APIs with Tanimoto similarity >= threshold
    num_apis = len(apis)
    adj = {i: [] for i in range(num_apis)}
    fps = apis["fp"].tolist()

    for i in range(num_apis):
        for j in range(i + 1, num_apis):
            if fps[i] and fps[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

    # 3. Use Connected Components to assign Cluster IDs
    cluster_ids = np.full(num_apis, -1)
    curr_cluster = 0
    for i in range(num_apis):
        if cluster_ids[i] == -1:
            stack = [i]
            while stack:
                node = stack.pop()
                if cluster_ids[node] == -1:
                    cluster_ids[node] = curr_cluster
                    stack.extend(adj[node])
            curr_cluster += 1

    apis["cluster_id"] = cluster_ids
    return df.merge(apis[["unii", "cluster_id"]], on="unii", how="left")


def split_by_api_cluster(df, seed=42):
    """
    Performs 80/10/10 split by cluster_id.
    Expects df to have 'unii' and 'smiles' columns (set during preprocessing).
    """
    # Ensure every row has a cluster_id
    print("  Assigning API clusters by chemical similarity...")
    df = assign_clusters(df)

    # Fill remaining (no SMILES) with their raw UNII as a fallback cluster
    df["cluster_id"] = df["cluster_id"].fillna(df["unii"])

    n_clusters = df["cluster_id"].nunique()
    print(f"  Found {n_clusters} API clusters")

    # --- Train (80%) vs Temp (20%) ---
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["cluster_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # --- Val (10%) vs Test (10%) ---
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["cluster_id"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    import pickle

    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    df = data["df"]
    train, val, test = split_by_api_cluster(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

"""
compute_features.py — One-time computation of molecular descriptors.

Run this ONCE to generate Data/api_features.csv.
After that, preprocess.py will load features from CSV instead of recomputing.

Usage:
    python compute_features.py
"""

import pandas as pd
import numpy as np
import json
from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import Descriptors


# ─────────────────────────────────────────────
# 20 MOLECULAR DESCRIPTORS (matches api_in=20)
# ─────────────────────────────────────────────
DESCRIPTOR_FUNCS = OrderedDict([
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("MolMR", Descriptors.MolMR),
    ("LabuteASA", Descriptors.LabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("NHOHCount", Descriptors.NHOHCount),
])

DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCS.keys())


def compute_descriptors(smiles):
    """Compute 20 molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            val = func(mol)
            if val is None or np.isinf(val) or np.isnan(val):
                val = 0.0
            desc[name] = float(val)
        except Exception:
            desc[name] = 0.0
    return desc


def main():
    print("=" * 50)
    print("Computing API Molecular Descriptors")
    print("=" * 50)

    # Load raw data
    df = pd.read_csv("f3.csv")
    print(f"Loaded {len(df)} rows")

    # Extract unique SMILES per API (avoids recomputing for same molecule)
    def extract_smiles(json_str):
        try:
            smap = json.loads(json_str)
            return list(smap.values())[0]
        except Exception:
            return None

    df["smiles"] = df["api_smiles_map"].apply(extract_smiles)
    unique_apis = df[["api_unii", "smiles"]].drop_duplicates(subset="api_unii").dropna(subset=["smiles"])
    print(f"Unique APIs with SMILES: {len(unique_apis)}")

    # Compute descriptors per unique API
    print("Computing descriptors (this is the slow part)...")
    results = []
    failed = 0

    for idx, (_, row) in enumerate(unique_apis.iterrows()):
        desc = compute_descriptors(row["smiles"])
        if desc is None:
            failed += 1
            continue

        desc["api_unii"] = row["api_unii"]
        desc["smiles"] = row["smiles"]
        results.append(desc)

        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(unique_apis)} APIs processed...")

    print(f"  Done. Computed: {len(results)}, Failed: {failed}")

    # Save to CSV
    features_df = pd.DataFrame(results)

    # Reorder columns: api_unii, smiles, then descriptors
    cols = ["api_unii", "smiles"] + DESCRIPTOR_NAMES
    features_df = features_df[cols]

    output_path = "api_features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"\n[OK] Saved {len(features_df)} API feature vectors to {output_path}")
    print(f"     Columns: {list(features_df.columns)}")


if __name__ == "__main__":
    main()


pip install rdkit torch_geometric

pip install rdkit torch_geometric

"""
build_graph.py — Builds a heterogeneous graph for ExciPick HetGNN.

Graph structure:
  Node types:
    - api:       molecular descriptor features (20-dim)
    - excipient: learnable embeddings (no initial features)

  Edge types:
    - (api, uses, excipient):           from training formulations
    - (excipient, used_by, api):        reverse of above
    - (excipient, cooccurs, excipient): Jaccard similarity >= threshold
    - (api, similar, api):              Tanimoto similarity >= threshold

Usage:
    python build_graph.py
"""

import pickle
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import HeteroData

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# from split import split_by_api_cluster
# from config import CONFIG


def get_morgan_fp(smiles):
    """Generates 2048-bit Morgan Fingerprint (radius 2)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None


def compute_jaccard_edges(train_df, num_excipients, threshold):
    """
    Compute excipient co-occurrence edges using Jaccard similarity.

    Jaccard(A, B) = |formulations with both A and B| / |formulations with A or B|
    """
    print(f"  Computing Jaccard co-occurrence edges (threshold={threshold})...")

    # For each excipient, track which formulations contain it
    exc_formulations = defaultdict(set)
    for idx, row in train_df.iterrows():
        for eid in row["excipient_ids"]:
            exc_formulations[eid].add(idx)

    # Only compute Jaccard for excipients that actually appear
    exc_ids = sorted(exc_formulations.keys())
    n = len(exc_ids)

    src, dst = [], []
    computed = 0

    for i in range(n):
        set_a = exc_formulations[exc_ids[i]]
        for j in range(i + 1, n):
            set_b = exc_formulations[exc_ids[j]]

            intersection = len(set_a & set_b)
            if intersection == 0:
                continue

            union = len(set_a | set_b)
            jaccard = intersection / union

            if jaccard >= threshold:
                src.extend([exc_ids[i], exc_ids[j]])  # bidirectional
                dst.extend([exc_ids[j], exc_ids[i]])
                computed += 1

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n} excipients processed...")

    print(f"  Found {computed} co-occurrence pairs ({len(src)} directed edges)")

    if src:
        return torch.tensor([src, dst], dtype=torch.long)
    return torch.zeros((2, 0), dtype=torch.long)


def compute_similarity_edges(unique_apis, threshold):
    """
    Compute API-API similarity edges using Tanimoto on Morgan fingerprints.
    """
    print(f"  Computing API similarity edges (Tanimoto >= {threshold})...")

    smiles_list = unique_apis["smiles"].tolist()
    fps = [get_morgan_fp(s) if isinstance(s, str) else None for s in smiles_list]

    n = len(fps)
    src, dst = [], []
    pairs = 0

    for i in range(n):
        if fps[i] is None:
            continue
        for j in range(i + 1, n):
            if fps[j] is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim >= threshold:
                src.extend([i, j])  # bidirectional
                dst.extend([j, i])
                pairs += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} APIs processed...")

    print(f"  Found {pairs} similar API pairs ({len(src)} directed edges)")

    if src:
        return torch.tensor([src, dst], dtype=torch.long)
    return torch.zeros((2, 0), dtype=torch.long)


def main():
    print("=" * 50)
    print("Building Heterogeneous Graph")
    print("=" * 50)

    # 1. Load preprocessed data
    print("\n[1] Loading preprocessed data...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    df = data["df"]
    excipient_vocab = data["excipient_vocab"]
    vocab_size = len(excipient_vocab)

    # 2. Split to get training data (edges only from train split)
    print("\n[2] Splitting data...")
    train_df, val_df, test_df = split_by_api_cluster(df, seed=CONFIG["seed"])
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Build API nodes — ALL unique APIs (train + val + test)
    print("\n[3] Building API nodes...")
    unique_apis = df[["api_unii", "smiles"]].drop_duplicates(subset="api_unii").reset_index(drop=True)

    # Get features for each unique API (take first row per api_unii)
    api_features_map = {}
    for _, row in df.iterrows():
        if row["api_unii"] not in api_features_map:
            api_features_map[row["api_unii"]] = row["api_features"]

    # Ensure consistent ordering
    unique_apis = unique_apis.reset_index(drop=True)
    api_unii_to_idx = {unii: i for i, unii in enumerate(unique_apis["api_unii"])}

    api_features = torch.tensor(
        [api_features_map[unii] for unii in unique_apis["api_unii"]],
        dtype=torch.float32
    )
    print(f"  API nodes: {len(api_unii_to_idx)}, features shape: {api_features.shape}")

    # 4. Build (api, uses, excipient) edges — TRAINING DATA ONLY
    print("\n[4] Building 'uses' edges from training data...")
    uses_set = set()
    for _, row in train_df.iterrows():
        api_idx = api_unii_to_idx[row["api_unii"]]
        for eid in row["excipient_ids"]:
            uses_set.add((api_idx, eid))

    uses_src = [e[0] for e in uses_set]
    uses_dst = [e[1] for e in uses_set]
    uses_edge = torch.tensor([uses_src, uses_dst], dtype=torch.long)
    used_by_edge = torch.tensor([uses_dst, uses_src], dtype=torch.long)
    print(f"  'uses' edges: {len(uses_set)} unique (api → excipient)")

    # 5. Build (excipient, cooccurs, excipient) edges — Jaccard
    print("\n[5] Building co-occurrence edges (Jaccard)...")
    cooccur_edge = compute_jaccard_edges(
        train_df, vocab_size, CONFIG["jaccard_threshold"]
    )

    # 6. Build (api, similar, api) edges — Tanimoto
    print("\n[6] Building API similarity edges (Tanimoto)...")
    similar_edge = compute_similarity_edges(
        unique_apis, CONFIG["similarity_threshold"]
    )

    # 7. Assemble HeteroData
    print("\n[7] Assembling HeteroData...")
    graph = HeteroData()

    # Node features
    graph["api"].x = api_features
    graph["api"].num_nodes = len(api_unii_to_idx)
    graph["excipient"].num_nodes = vocab_size

    # Edge indices
    graph["api", "uses", "excipient"].edge_index = uses_edge
    graph["excipient", "used_by", "api"].edge_index = used_by_edge
    graph["excipient", "cooccurs", "excipient"].edge_index = cooccur_edge
    graph["api", "similar", "api"].edge_index = similar_edge

    print(f"\n  Graph summary:")
    print(f"    API nodes:        {graph['api'].num_nodes}")
    print(f"    Excipient nodes:  {graph['excipient'].num_nodes}")
    print(f"    uses edges:       {uses_edge.shape[1]}")
    print(f"    used_by edges:    {used_by_edge.shape[1]}")
    print(f"    cooccurs edges:   {cooccur_edge.shape[1]}")
    print(f"    similar edges:    {similar_edge.shape[1]}")

    # 8. Save
    torch.save(graph, "hetero_graph.pt")
    print(f"\n  Saved graph to: hetero_graph.pt")

    with open("api_node_mapping.pkl", "wb") as f:
        pickle.dump(api_unii_to_idx, f)
    print(f"  Saved API mapping to: api_node_mapping.pkl")

    # Also save split indices for training.py to reuse
    with open("split_data.pkl", "wb") as f:
        pickle.dump({
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
        }, f)
    print(f"  Saved split data to: split_data.pkl")

    print(f"\n{'=' * 50}")
    print("[OK] Graph build complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()


"""
dataset.py — PyTorch Dataset for ExciPick HetGNN model.

Each sample returns:
  - api_idx:  scalar long — index of the API node in the heterogeneous graph
  - dose:     scalar float — normalized log dose
  - per_unit: scalar long — per-unit ID
  - route:    scalar long — route ID
  - form:     scalar long — dosage form ID
  - target:   (vocab_size,) float tensor — multi-hot excipient labels
"""

import torch
from torch.utils.data import Dataset


class ExciDataset(Dataset):
    def __init__(self, df, excipient_vocab_size, api_node_mapping):
        """
        Args:
            df: preprocessed DataFrame with columns:
                api_unii, dose_normalized, per_unit_id, route_id, form_id, excipient_ids
            excipient_vocab_size: total number of excipients in vocabulary
            api_node_mapping: dict mapping api_unii -> graph node index
        """
        self.df = df.reset_index(drop=True)
        self.vocab_size = excipient_vocab_size
        self.api_node_mapping = api_node_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # API node index in the heterogeneous graph
        api_idx = torch.tensor(
            self.api_node_mapping[row["api_unii"]], dtype=torch.long
        )

        # Normalized dose
        dose = torch.tensor(row["dose_normalized"], dtype=torch.float32)

        # Categorical IDs
        per_unit = torch.tensor(row["per_unit_id"], dtype=torch.long)
        route = torch.tensor(row["route_id"], dtype=torch.long)
        form = torch.tensor(row["form_id"], dtype=torch.long)

        # Multi-hot target over excipient vocab
        target = torch.zeros(self.vocab_size, dtype=torch.float32)
        for eid in row["excipient_ids"]:
            target[eid] = 1.0

        return {
            "api_idx": api_idx,
            "dose": dose,
            "per_unit": per_unit,
            "route": route,
            "form": form,
            "target": target,
        }


"""
metrics.py — Evaluation metrics for ExciPick.

Metrics:
  - Precision@K: fraction of top-K predictions that are correct
  - Recall@K: fraction of true excipients captured in top-K
  - F1@K: harmonic mean of Precision@K and Recall@K
  - Jaccard@K: intersection / union of predicted and true sets
"""

import torch
import numpy as np


def precision_at_k(predictions, targets, k=10):
    """
    Args:
        predictions: (B, V) raw logits or scores
        targets: (B, V) multi-hot binary targets
        k: number of top predictions to consider
    Returns:
        mean precision@k across batch
    """
    _, topk_idx = predictions.topk(k, dim=1)
    topk_hits = targets.gather(1, topk_idx)
    return topk_hits.sum(dim=1).float().div(k).mean().item()


def recall_at_k(predictions, targets, k=10):
    """Mean recall@k across batch."""
    _, topk_idx = predictions.topk(k, dim=1)
    topk_hits = targets.gather(1, topk_idx)
    true_counts = targets.sum(dim=1).clamp(min=1)
    return (topk_hits.sum(dim=1).float() / true_counts).mean().item()


def f1_at_k(predictions, targets, k=10):
    """Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(predictions, targets, k)
    r = recall_at_k(predictions, targets, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def jaccard_at_k(predictions, targets, k=10):
    """
    Mean Jaccard similarity between top-K predicted set and true set.
    Jaccard = |intersection| / |union|
    """
    _, topk_idx = predictions.topk(k, dim=1)

    # Build predicted set as multi-hot
    pred_set = torch.zeros_like(targets)
    pred_set.scatter_(1, topk_idx, 1.0)

    intersection = (pred_set * targets).sum(dim=1)
    union = ((pred_set + targets) > 0).float().sum(dim=1).clamp(min=1)

    return (intersection / union).mean().item()


def evaluate(model, dataloader, device, k_values=None):
    """
    Run full evaluation on a DataLoader.

    Returns a dict of metrics for each k value.
    """
    if k_values is None:
        k_values = [5, 10, 15]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            api = batch["api"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            output = model(api, dose, per_unit, route, form)

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    results = {}
    for k in k_values:
        results[f"Precision@{k}"] = precision_at_k(preds, targets, k)
        results[f"Recall@{k}"] = recall_at_k(preds, targets, k)
        results[f"F1@{k}"] = f1_at_k(preds, targets, k)
        results[f"Jaccard@{k}"] = jaccard_at_k(preds, targets, k)

    return results


def print_metrics(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Group by k
    k_values = sorted(set(int(key.split("@")[1]) for key in results))
    metrics = ["Precision", "Recall", "F1", "Jaccard"]

    # Header
    header = f"{'Metric':<12}" + "".join(f"{'@'+str(k):>10}" for k in k_values)
    print(header)
    print("-" * len(header))

    for metric in metrics:
        row = f"{metric:<12}"
        for k in k_values:
            val = results.get(f"{metric}@{k}", 0)
            row += f"{val:>10.4f}"
        print(row)

    print("=" * 50)


"""
api_encoder.py — API feature projector for ExciPick HetGNN.

Projects 20 molecular descriptors to GNN hidden dimension.
The heavy representational lifting is done by the GNN layers.
"""

import torch.nn as nn


class APIProjector(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(CONFIG["api_in"], CONFIG["gnn_hidden"]),
            nn.LayerNorm(CONFIG["gnn_hidden"]),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
import torch
import torch.nn as nn


class StrengthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.per_unit_emb = nn.Embedding(
            CONFIG["per_unit_vocab"],
            CONFIG["per_unit_emb"]
        )

        self.net = nn.Sequential(
            nn.Linear(1 + CONFIG["per_unit_emb"], CONFIG["strength_out"]),
            nn.ReLU()
        )

    def forward(self, dose, per_unit):
        emb = self.per_unit_emb(per_unit)
        x = torch.cat([dose.unsqueeze(1), emb], dim=1)
        return self.net(x)
"""
gnn_layers.py — Heterogeneous GNN encoder for ExciPick.

Uses PyG's HeteroConv with SAGEConv per edge type.
Each layer: HeteroConv → LayerNorm → ReLU → Dropout (+ residual).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNNEncoder(nn.Module):
    """
    Multi-layer heterogeneous GNN using SAGEConv per edge type.

    Args:
        metadata: tuple of (node_types, edge_types) from HeteroData.metadata()
        hidden_dim: hidden dimension for all layers
        num_layers: number of message passing rounds
        dropout: dropout rate
    """

    def __init__(self, metadata, hidden_dim, num_layers, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # One SAGEConv per edge type
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            # LayerNorm per node type
            norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in metadata[0]
            })
            self.norms.append(norm_dict)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: {node_type: (num_nodes, hidden_dim)} node features
            edge_index_dict: {edge_type: (2, num_edges)} edge indices

        Returns:
            x_dict: enriched node embeddings, same structure as input
        """
        for conv, norm_dict in zip(self.convs, self.norms):
            # Message passing
            x_dict_new = conv(x_dict, edge_index_dict)

            # Residual + LayerNorm + ReLU + Dropout
            x_dict = {
                key: F.dropout(
                    F.relu(norm_dict[key](x_dict_new[key] + x_dict[key])),
                    p=self.dropout,
                    training=self.training,
                )
                for key in x_dict_new.keys()
            }

        return x_dict

"""
excipient_scorer.py — Scores each excipient against a formulation context.

In the HetGNN version, excipient embeddings come from the GNN encoder
(enriched via message passing), not from a local nn.Embedding.
"""

import torch
import torch.nn as nn


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()

        # Bilinear interaction: context (192) x excipient (128) -> 1 logit
        self.bilinear = nn.Bilinear(CONFIG["context_out"], CONFIG["gnn_hidden"], 1)

    def forward(self, context, exc_embs):
        """
        Args:
            context:  (B, context_out) — fused formulation context
            exc_embs: (V, gnn_hidden)  — GNN-enriched excipient embeddings

        Returns:
            scores: (B, V) — one logit per excipient
        """
        B = context.shape[0]
        V = exc_embs.shape[0]

        # Tile context and excipient embeddings for pairwise scoring
        context_exp = context.unsqueeze(1).expand(-1, V, -1)   # (B, V, context_out)
        exc_exp = exc_embs.unsqueeze(0).expand(B, -1, -1)      # (B, V, gnn_hidden)

        # Bilinear expects (B*V, in1), (B*V, in2)
        scores = self.bilinear(context_exp.reshape(-1, CONFIG["context_out"]),
                               exc_exp.reshape(-1, CONFIG["gnn_hidden"]))
        return scores.view(B, V)
"""
FULL_MODEL.py — ExciPick Heterogeneous GNN model.

Architecture:
  1. Project API features + excipient embeddings to GNN hidden dim
  2. HetGNN message passing enriches all node embeddings
  3. Per-sample: look up enriched API embedding + encode dose/route/form
  4. Score enriched context against all enriched excipient embeddings
"""

import torch
import torch.nn as nn



class ExciPickHGNN(nn.Module):
    def __init__(self, graph_metadata, vocab_size):
        """
        Args:
            graph_metadata: tuple from HeteroData.metadata()
                            (node_types, edge_types)
            vocab_size: number of excipients in vocabulary
        """
        super().__init__()

        # --- Node feature projectors ---
        self.api_proj = APIProjector()
        self.exc_emb = nn.Embedding(vocab_size, CONFIG["gnn_hidden"])

        # --- GNN encoder ---
        self.gnn = HeteroGNNEncoder(
            metadata=graph_metadata,
            hidden_dim=CONFIG["gnn_hidden"],
            num_layers=CONFIG["gnn_layers"],
            dropout=CONFIG["gnn_dropout"],
        )

        # --- Dose encoder (unchanged from v1) ---
        self.strength_encoder = StrengthEncoder()

        # --- Route / Form embeddings ---
        self.route_emb = nn.Embedding(CONFIG["route_vocab"], CONFIG["route_emb"])
        self.form_emb = nn.Embedding(CONFIG["form_vocab"], CONFIG["form_emb"])

        # --- Context fusion ---
        fusion_in = (
            CONFIG["gnn_hidden"]
            + CONFIG["strength_out"]
            + CONFIG["route_emb"]
            + CONFIG["form_emb"]
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, CONFIG["context_out"]),
            nn.LayerNorm(CONFIG["context_out"]),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout_context"]),
        )

        # --- Excipient scorer ---
        self.scorer = Scorer()

    def forward(self, graph, api_idx, dose, per_unit, route, form):
        """
        Args:
            graph:    HeteroData with enriched node features
            api_idx:  (B,) long — indices of API nodes for this batch
            dose:     (B,) float — normalized dose
            per_unit: (B,) long — per-unit category ID
            route:    (B,) long — route category ID
            form:     (B,) long — dosage form category ID

        Returns:
            scores: (B, vocab_size) — raw logit scores per excipient
        """
        # 1. Project node features to GNN hidden dim
        x_dict = {
            "api": self.api_proj(graph["api"].x),
            "excipient": self.exc_emb.weight,
        }

        # 2. GNN message passing
        enriched = self.gnn(x_dict, graph.edge_index_dict)
        enriched_api = enriched["api"]           # (num_apis, gnn_hidden)
        enriched_exc = enriched["excipient"]     # (V, gnn_hidden)

        # 3. Look up this batch's API embeddings
        batch_api = enriched_api[api_idx]        # (B, gnn_hidden)

        # 4. Encode dose strength
        strength = self.strength_encoder(dose, per_unit)   # (B, strength_out)

        # 5. Encode route and form
        route_e = self.route_emb(route)          # (B, route_emb)
        form_e = self.form_emb(form)             # (B, form_emb)

        # 6. Fuse all context
        context = self.fusion(
            torch.cat([batch_api, strength, route_e, form_e], dim=1)
        )  # (B, context_out)

        # 7. Score against all enriched excipient embeddings
        scores = self.scorer(context, enriched_exc)   # (B, V)

        return scores

"""
training.py — Full training pipeline for ExciPick HetGNN.

Pipeline:
  1. Load preprocessed data + heterogeneous graph + split
  2. Create DataLoaders
  3. Train with GNN-aware forward pass + validation loop
  4. Save best model
"""

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# from dataset import ExciDataset
# from model.FULL_MODEL import ExciPickHGNN
# from config import CONFIG
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def main():
    # ─────────────────────────────────────────
    # SEED
    # ─────────────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ─────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────
    print("Loading preprocessed data...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    excipient_vocab = data["excipient_vocab"]
    vocab_size = len(excipient_vocab)

    print("Loading split data...")
    with open("split_data.pkl", "rb") as f:
        split_data = pickle.load(f)

    train_df = split_data["train_df"]
    val_df = split_data["val_df"]
    test_df = split_data["test_df"]

    print(f"  Dataset splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Excipient vocab: {vocab_size}")

    # ─────────────────────────────────────────
    # LOAD GRAPH + API MAPPING
    # ─────────────────────────────────────────
    print("\nLoading heterogeneous graph...")
    graph = torch.load("hetero_graph.pt", weights_only=False)
    print(f"  API nodes: {graph['api'].num_nodes}")
    print(f"  Excipient nodes: {graph['excipient'].num_nodes}")

    with open("api_node_mapping.pkl", "rb") as f:
        api_node_mapping = pickle.load(f)

    # ─────────────────────────────────────────
    # DATASETS + DATALOADERS
    # ─────────────────────────────────────────
    train_dataset = ExciDataset(train_df, vocab_size, api_node_mapping)
    val_dataset = ExciDataset(val_df, vocab_size, api_node_mapping)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    # ─────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Move graph to device
    graph = graph.to(device)

    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    ).to(device)

    # Dummy forward pass to initialize lazy SAGEConv parameters
    with torch.no_grad():
        dummy_idx = torch.zeros(1, dtype=torch.long, device=device)
        dummy_dose = torch.zeros(1, device=device)
        dummy_cat = torch.zeros(1, dtype=torch.long, device=device)
        model(graph, dummy_idx, dummy_dose, dummy_cat, dummy_cat, dummy_cat)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
    )

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )

    # Class imbalance is handled dynamically by Focal Loss
    # We no longer need the extreme pos_weight approach
    print("  Using Focal Loss (alpha=0.25, gamma=2.0) instead of standard BCE.")

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # ─────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\nTraining for {CONFIG['epochs']} epochs...\n")

    for epoch in range(CONFIG["epochs"]):

        # --- Train ---
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False):
            api_idx = batch["api_idx"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(graph, api_idx, dose, per_unit, route, form)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", leave=False):
                api_idx = batch["api_idx"].to(device)
                dose = batch["dose"].to(device)
                per_unit = batch["per_unit"].to(device)
                route = batch["route"].to(device)
                form = batch["form"].to(device)
                target = batch["target"].to(device)

                output = model(graph, api_idx, dose, per_unit, route, form)
                loss = loss_fn(output, target)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Logging ---
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']}  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"LR: {current_lr:.6f}", end="")

        # --- Save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("  * saved", end="")

        print()

    print(f"\n[OK] Training complete. Best val loss: {best_val_loss:.4f}")
    print("   Model saved to: best_model.pt")

    # ─────────────────────────────────────────
    # SAVE TEST SET FOR EVALUATION
    # ─────────────────────────────────────────
    with open("test_data.pkl", "wb") as f:
        pickle.dump({
            "test_df": test_df,
            "excipient_vocab": excipient_vocab,
            "vocab_size": vocab_size,
        }, f)
    print("   Test data saved to: test_data.pkl")


if __name__ == "__main__":
    main()

pip install torch_geometric

"""
test.py — Evaluate the trained ExciPick HetGNN model on the test set.

Usage:
  python test.py                     # evaluate saved model
  python test.py --smoke             # quick smoke test with dummy data
"""

import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

# from model.FULL_MODEL import ExciPickHGNN
# from dataset import ExciDataset
# from metrics import evaluate, print_metrics
# from config import CONFIG


def smoke_test():
    """Quick forward-pass check with dummy graph and data."""
    print("Running smoke test...")

    vocab_size = 100

    # Build a minimal dummy graph
    graph = HeteroData()
    graph["api"].x = torch.randn(10, CONFIG["api_in"])
    graph["api"].num_nodes = 10
    graph["excipient"].num_nodes = vocab_size

    graph["api", "uses", "excipient"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    graph["excipient", "used_by", "api"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    graph["excipient", "cooccurs", "excipient"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )
    graph["api", "similar", "api"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )

    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    )

    api_idx = torch.randint(0, 10, (4,))
    dose = torch.randn(4)
    per_unit = torch.randint(0, 3, (4,))
    route = torch.randint(0, 10, (4,))
    form = torch.randint(0, 10, (4,))

    output = model(graph, api_idx, dose, per_unit, route, form)
    print(f"[OK] Smoke test passed! Output shape: {output.shape}")
    assert output.shape == (4, vocab_size), f"Expected (4, {vocab_size}), got {output.shape}"


def run_evaluation():
    """Load best model + graph + test data and compute metrics."""
    print("Loading test data...")
    with open("test_data (1).pkl", "rb") as f:
        test_data = pickle.load(f)

    test_df = test_data["test_df"]
    excipient_vocab = test_data["excipient_vocab"]
    vocab_size = test_data["vocab_size"]

    print(f"  Test set: {len(test_df)} rows")
    print(f"  Vocab size: {vocab_size}")

    # Load API node mapping
    with open("api_node_mapping (1).pkl", "rb") as f:
        api_node_mapping = pickle.load(f)

    # Dataset + Loader
    test_dataset = ExciDataset(test_df, vocab_size, api_node_mapping)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )

    # Load graph
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    graph = torch.load("hetero_graph (1).pt", map_location=device, weights_only=False)
    graph = graph.to(device)

    # Load model
    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    ).to(device)
    model.load_state_dict(torch.load("best_model (1).pt", map_location=device))
    print(f"  Loaded best_model.pt on {device}")

    # Evaluate using graph-aware forward pass
    k_values = [5, CONFIG["top_k"], 15]
    results = evaluate_with_graph(model, graph, test_loader, device, k_values)
    print_metrics(results)


def evaluate_with_graph(model, graph, dataloader, device, k_values=None):
    """Run full evaluation with graph-aware forward pass."""
    # from metrics import precision_at_k, recall_at_k, f1_at_k, jaccard_at_k

    if k_values is None:
        k_values = [5, 10, 15]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            api_idx = batch["api_idx"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            output = model(graph, api_idx, dose, per_unit, route, form)

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    results = {}
    for k in k_values:
        results[f"Precision@{k}"] = precision_at_k(preds, targets, k)
        results[f"Recall@{k}"] = recall_at_k(preds, targets, k)
        results[f"F1@{k}"] = f1_at_k(preds, targets, k)
        results[f"Jaccard@{k}"] = jaccard_at_k(preds, targets, k)

    return results




run_evaluation()

ls

"""
predict.py — Inference script for ExciPick HetGNN.

Given an API (by UNII), dose, unit, route, and form,
predicts the most likely excipients for the formulation.

Usage:
    python predict.py --api 6CW7F3G59X --dose 500 --unit 1 --route ORAL --form TABLET
"""

import torch
import numpy as np
import pickle
import argparse
# from config import CONFIG
# from model.FULL_MODEL import ExciPickHGNN


def load_metadata():
    print("Loading metadata...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def predict_excipients(
    model, graph, data, api_node_mapping, api_unii, dose_mg, unit, route, form, threshold=0.5
):
    device = next(model.parameters()).device
    model.eval()

    # 1. Get Vocabularies & Stats
    excipient_vocab = data["excipient_vocab"]
    route_vocab = data["route_vocab"]
    form_vocab = data["form_vocab"]
    per_unit_vocab = data["per_unit_vocab"]
    dose_stats = data["dose_stats"]

    # Reverse vocab to map IDs back to string names
    id_to_excipient = {v: k for k, v in excipient_vocab.items()}

    # 2. Look up API node index in graph
    if api_unii not in api_node_mapping:
        valid_examples = list(api_node_mapping.keys())[:3]
        raise ValueError(
            f"API UNII '{api_unii}' not found in the graph.\n"
            f"Some valid examples: {valid_examples}"
        )

    api_idx = torch.tensor([api_node_mapping[api_unii]], dtype=torch.long).to(device)

    # 3. Process Dose
    log_dose = np.log10(max(dose_mg, 1e-9))
    dose_normalized = (log_dose - dose_stats["mean"]) / dose_stats["std"]
    dose_tensor = torch.tensor([dose_normalized], dtype=torch.float32).to(device)

    # 4. Process Categorical IDs
    if unit not in per_unit_vocab:
        raise ValueError(f"Unit '{unit}' not found. Valid units: {list(per_unit_vocab.keys())}")
    if route not in route_vocab:
        raise ValueError(f"Route '{route}' not found. Valid routes: {list(route_vocab.keys())}")
    if form not in form_vocab:
        raise ValueError(f"Form '{form}' not found. Valid forms: {list(form_vocab.keys())[:10]}...")

    unit_tensor = torch.tensor([per_unit_vocab[unit]], dtype=torch.long).to(device)
    route_tensor = torch.tensor([route_vocab[route]], dtype=torch.long).to(device)
    form_tensor = torch.tensor([form_vocab[form]], dtype=torch.long).to(device)

    # 5. Model Forward Pass (with graph)
    with torch.no_grad():
        logits = model(graph, api_idx, dose_tensor, unit_tensor, route_tensor, form_tensor)
        probs = torch.sigmoid(logits)[0]

    # 6. Extract predictions above threshold
    predictions = []
    for exc_id in range(len(id_to_excipient)):
        prob = probs[exc_id].item()
        if prob >= threshold:
            predictions.append((id_to_excipient[exc_id], prob))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick HetGNN Inference")
    parser.add_argument("--api", type=str, default="6CW7F3G59X", help="API UNII code")
    parser.add_argument("--dose", type=float, default=100.0, help="Dose in mg")
    parser.add_argument("--unit", type=str, default="1", help="Denominator unit")
    parser.add_argument("--route", type=str, default="ORAL", help="Route of administration")
    parser.add_argument("--form", type=str, default="TABLET", help="Dosage form")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Model weights path")

    args = parser.parse_args()

    # Load metadata
    try:
        data = load_metadata()
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run preprocess.py first.")
        exit(1)

    vocab_size = len(data["excipient_vocab"])

    # Load graph
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        graph = torch.load("hetero_graph.pt", map_location=device, weights_only=False)
        graph = graph.to(device)
    except FileNotFoundError:
        print("Error: hetero_graph.pt not found. Run build_graph.py first.")
        exit(1)

    # Load API node mapping
    with open("api_node_mapping.pkl", "rb") as f:
        api_node_mapping = pickle.load(f)

    # Load model
    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Successfully loaded model weights from {args.model}")
    except FileNotFoundError:
        print(f"Warning: '{args.model}' not found. Using random weights for demo!")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        exit(1)

    print("\n" + "=" * 50)
    print("INPUTS:")
    print(f"  API UNII : {args.api}")
    print(f"  Dose     : {args.dose} mg")
    print(f"  Unit     : {args.unit}")
    print(f"  Route    : {args.route}")
    print(f"  Form     : {args.form}")
    print("=" * 50 + "\n")

    try:
        predictions = predict_excipients(
            model=model,
            graph=graph,
            data=data,
            api_node_mapping=api_node_mapping,
            api_unii=args.api,
            dose_mg=args.dose,
            unit=args.unit,
            route=args.route,
            form=args.form,
            threshold=args.threshold,
        )

        print(f"PREDICTED EXCIPIENTS (Threshold >= {args.threshold}):")
        if not predictions:
            print("  (No excipients predicted above the threshold)")
        else:
            for name, prob in predictions:
                print(f"  - {name:<30} ({prob:.1%} confidence)")

    except ValueError as e:
        print(f"Input Error: {e}")


