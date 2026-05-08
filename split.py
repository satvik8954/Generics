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
        fpg = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return fpg.GetFingerprint(mol)
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