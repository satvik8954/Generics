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

from split import split_by_api_cluster
from config import CONFIG


def get_morgan_fp(smiles):
    """Generates 2048-bit Morgan Fingerprint (radius 2)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fpg = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return fpg.GetFingerprint(mol)
    return None


def compute_jaccard_edges(train_df, num_excipients, threshold):
    """
    Compute excipient co-occurrence edges using Jaccard similarity.
    Returns edge_index AND edge_weight (Jaccard values).

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

    src, dst, weights = [], [], []
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
                weights.extend([jaccard, jaccard])     # same weight both directions
                computed += 1

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n} excipients processed...")

    print(f"  Found {computed} co-occurrence pairs ({len(src)} directed edges)")

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weight
    return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)


def compute_similarity_edges(unique_apis, threshold):
    """
    Compute API-API similarity edges using Tanimoto on Morgan fingerprints.
    Returns edge_index AND edge_weight (Tanimoto values).
    """
    print(f"  Computing API similarity edges (Tanimoto >= {threshold})...")

    smiles_list = unique_apis["smiles"].tolist()
    fps = [get_morgan_fp(s) if isinstance(s, str) else None for s in smiles_list]

    n = len(fps)
    src, dst, weights = [], [], []
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
                weights.extend([sim, sim])  # same weight both directions
                pairs += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} APIs processed...")

    print(f"  Found {pairs} similar API pairs ({len(src)} directed edges)")

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weight
    return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)


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
        np.array([api_features_map[unii] for unii in unique_apis["api_unii"]]),
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
    cooccur_edge, cooccur_weight = compute_jaccard_edges(
        train_df, vocab_size, CONFIG["jaccard_threshold"]
    )

    # 6. Build (api, similar, api) edges — Tanimoto
    print("\n[6] Building API similarity edges (Tanimoto)...")
    similar_edge, similar_weight = compute_similarity_edges(
        unique_apis, CONFIG["similarity_threshold"]
    )

    # 7. Assemble HeteroData
    print("\n[7] Assembling HeteroData...")
    graph = HeteroData()

    # Node features
    graph["api"].x = api_features
    graph["api"].num_nodes = len(api_unii_to_idx)
    graph["excipient"].num_nodes = vocab_size

    # Edge indices + weights
    # uses/used_by: uniform weight = 1.0
    graph["api", "uses", "excipient"].edge_index = uses_edge
    graph["api", "uses", "excipient"].edge_weight = torch.ones(uses_edge.shape[1], dtype=torch.float32)
    graph["excipient", "used_by", "api"].edge_index = used_by_edge
    graph["excipient", "used_by", "api"].edge_weight = torch.ones(used_by_edge.shape[1], dtype=torch.float32)

    # cooccurs: weighted by Jaccard similarity
    graph["excipient", "cooccurs", "excipient"].edge_index = cooccur_edge
    graph["excipient", "cooccurs", "excipient"].edge_weight = cooccur_weight

    # similar: weighted by Tanimoto similarity
    graph["api", "similar", "api"].edge_index = similar_edge
    graph["api", "similar", "api"].edge_weight = similar_weight

    print(f"\n  Graph summary:")
    print(f"    API nodes:        {graph['api'].num_nodes}")
    print(f"    Excipient nodes:  {graph['excipient'].num_nodes}")
    print(f"    uses edges:       {uses_edge.shape[1]}")
    print(f"    used_by edges:    {used_by_edge.shape[1]}")
    print(f"    cooccurs edges:   {cooccur_edge.shape[1]}  (weighted by Jaccard)")
    print(f"    similar edges:    {similar_edge.shape[1]}  (weighted by Tanimoto)")
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
