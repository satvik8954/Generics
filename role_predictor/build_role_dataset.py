"""
build_role_dataset.py — Builds role prediction dataset from processed_data.pkl

Reads existing preprocessed formulations, maps excipient UNIIs to functional
roles, and builds a binary multi-label target vector per formulation.

Output: role_dataset.pkl
    {
        "X_train": np.array (N, 23),   # api_features + dose + route + form
        "y_train": np.array (N, 13),   # binary role targets
        "X_val":   ...,
        "y_val":   ...,
        "X_test":  ...,
        "y_test":  ...,
        "role_names": [list of 13 role names],
        "role_counts": {role: count in train set},
    }

Usage:
    python build_role_dataset.py
"""

import pickle
import json
import numpy as np
import pandas as pd
from collections import Counter

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROCESSED_DATA_PATH = "processed_data.pkl"
SPLIT_DATA_PATH     = "split_data.pkl"
ROLES_CSV_PATH      = "Data/excipient_roles_clean_final.csv"
OUTPUT_PATH         = "role_predictor/role_dataset.pkl"

# ─────────────────────────────────────────────
# ROLE TAXONOMY
# Maps canonical role name → all aliases from oral_role column
# ─────────────────────────────────────────────
ROLE_TAXONOMY = {
    "binder":             ["binder"],
    "filler":             ["filler", "diluent"],
    "disintegrant":       ["disintegrant"],
    "lubricant":          ["lubricant"],
    "glidant":            ["glidant", "anticaking agent", "adsorbent"],
    "coating_agent":      ["coating agent"],
    "colorant":           ["colorant", "coloring agent", "pigment", "opacifier"],
    "controlled_release": ["controlled-release agent"],
    "solvent":            ["solvent"],
    "surfactant":         ["surfactant", "emulsifying agent"],
    "plasticizer":        ["plasticizer"],
    "alkalizing_agent":   ["alkalizing agent"],
    "humectant":          ["humectant"],
}

ROLE_NAMES = list(ROLE_TAXONOMY.keys())
NUM_ROLES  = len(ROLE_NAMES)

# Build reverse lookup: alias → canonical
ALIAS_TO_CANONICAL = {}
for canonical, aliases in ROLE_TAXONOMY.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower().strip()] = canonical


def build_unii_to_roles(roles_csv_path):
    """
    Build {unii: set_of_canonical_roles} from excipient_roles_clean_final.csv
    """
    df = pd.read_csv(roles_csv_path)
    unii_to_roles = {}

    for _, row in df.iterrows():
        unii = str(row.get("unii", "")).strip()
        oral_role = row.get("oral_role", "")

        if not unii or not isinstance(oral_role, str):
            continue

        roles = set()
        for raw_role in oral_role.split(";"):
            canonical = ALIAS_TO_CANONICAL.get(raw_role.lower().strip())
            if canonical:
                roles.add(canonical)

        if roles:
            unii_to_roles[unii] = roles

    print(f"  Built UNII→role lookup: {len(unii_to_roles)} excipients with known roles")
    return unii_to_roles


def get_role_vector(ingredients_str, unii_to_roles):
    """
    Parse inactive_ingredients JSON and return 13-dim binary role vector.
    """
    target = np.zeros(NUM_ROLES, dtype=np.float32)
    try:
        items = json.loads(ingredients_str)
        for item in items:
            unii = str(item.get("unii", "")).strip()
            roles = unii_to_roles.get(unii, set())
            for role in roles:
                if role in ROLE_NAMES:
                    target[ROLE_NAMES.index(role)] = 1.0
    except Exception:
        pass
    return target


def build_features(row):
    """
    Build 23-dim input vector: api_features(20) + dose(1) + route_id(1) + form_id(1)
    """
    api_feats = np.array(row["api_features"], dtype=np.float32)   # (20,)
    dose      = np.array([row["dose_normalized"]], dtype=np.float32)  # (1,)
    route     = np.array([row["route_id"]], dtype=np.float32)     # (1,)
    form      = np.array([row["form_id"]], dtype=np.float32)      # (1,)
    return np.concatenate([api_feats, dose, route, form])          # (23,)


def build_split(df_split, unii_to_roles):
    """
    Build X (features) and y (role targets) for a dataframe split.
    """
    X, y = [], []
    skipped = 0

    for _, row in df_split.iterrows():
        target = get_role_vector(row["inactive_ingredients"], unii_to_roles)

        # Skip formulations where no roles could be mapped
        if target.sum() == 0:
            skipped += 1
            continue

        features = build_features(row)
        X.append(features)
        y.append(target)

    print(f"    Built {len(X)} samples, skipped {skipped} (no roles mapped)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    import os
    os.makedirs("role_predictor", exist_ok=True)

    print("=" * 50)
    print("Building Role Prediction Dataset")
    print("=" * 50)

    # 1. Load data
    print("\n[1] Loading processed data...")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    print("\n[2] Loading split data...")
    with open(SPLIT_DATA_PATH, "rb") as f:
        split = pickle.load(f)

    train_df = split["train_df"]
    val_df   = split["val_df"]
    test_df  = split["test_df"]
    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Build UNII → roles lookup
    print("\n[3] Building UNII → role lookup...")
    unii_to_roles = build_unii_to_roles(ROLES_CSV_PATH)

    # 3. Build splits
    print("\n[4] Building train split...")
    X_train, y_train = build_split(train_df, unii_to_roles)

    print("\n[5] Building val split...")
    X_val, y_val = build_split(val_df, unii_to_roles)

    print("\n[6] Building test split...")
    X_test, y_test = build_split(test_df, unii_to_roles)

    # 4. Role statistics on train set
    role_counts = {
        ROLE_NAMES[i]: int(y_train[:, i].sum())
        for i in range(NUM_ROLES)
    }

    print(f"\n[7] Role distribution in train set:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = count / len(y_train) * 100
        print(f"    {role:<25} {count:>4} / {len(y_train)} ({pct:.1f}%)")

    # 5. Save
    dataset = {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_val":      X_val,
        "y_val":      y_val,
        "X_test":     X_test,
        "y_test":     y_test,
        "role_names": ROLE_NAMES,
        "role_counts": role_counts,
        "num_roles":  NUM_ROLES,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\n{'=' * 50}")
    print(f"[OK] Dataset saved to: {OUTPUT_PATH}")
    print(f"     Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"     Roles: {NUM_ROLES} → {ROLE_NAMES}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
