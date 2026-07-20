"""
build_role_dataset.py — Builds role prediction dataset from processed_data.pkl

Data source change (v2):
  Previously: Data/excipient_roles_clean_final.csv (semicolon string oral_role column)
  Now:        Data/functional_categories_excipients_FILLED_1.csv (one-hot encoded HPE categories)

  The filled CSV has no UNII column, so excipient_roles_clean_final.csv is still
  used as a UNII lookup only (name → unii). The role data itself comes entirely
  from the one-hot columns in the filled CSV.

Output: role_dataset.pkl
    {
        "X_train": np.array (N, 23),   # api_features + dose + route + form
        "y_train": np.array (N, 25),   # binary role targets
        ...
        "role_names": [list of 25 role names],
        "role_counts": {role: count in train set},
    }

Usage:
    python build_role_dataset.py
"""

import pickle
import json
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROCESSED_DATA_PATH = "processed_data.pkl"
SPLIT_DATA_PATH     = "split_data.pkl"

# NEW: one-hot encoded HPE categories — primary role source
FILLED_CSV_PATH     = "Data/functional_categories_excipients_FILLED_1.csv"

# Still used for UNII lookup only (name → unii mapping)
UNII_LOOKUP_PATH    = "Data/excipient_roles_clean_final.csv"

OUTPUT_PATH         = "role_predictor/role_dataset.pkl"

# Keep as alias so predict_roles.py import still works
ROLES_CSV_PATH = FILLED_CSV_PATH

# ─────────────────────────────────────────────
# ROLE TAXONOMY
# Maps canonical role → HPE category column names in filled CSV
# Case must match the actual column headers in the CSV exactly
# ─────────────────────────────────────────────
ROLE_TAXONOMY = {
    "binder":                      ["Tablet binder", "Binding agent"],
    "filler":                      ["Tablet and capsule diluent", "Diluent", "Tablet diluent",
                                     "Tablet filler", "Tablet and capsule filler", "Bulking agent",
                                     "Directly compressible tablet excipient"],
    "disintegrant":                ["Tablet and capsule disintegrant", "Tablet disintegrant",
                                     "Disintegrant", "Water-absorbing agent"],
    "lubricant":                   ["Tablet and capsule lubricant", "Lubricant", "Tablet lubricant",
                                     "Antiadherent"],
    "glidant":                     ["Glidant", "Anticaking agent", "Adsorbent"],
    "coating_agent":               ["Coating agent", "Film-forming agent", "Enteric coating agent",
                                     "Membrane-forming agent", "Polishing agent", "Encapsulating agent",
                                     "Opacifier"],
    "controlled_release":          ["Controlled-release agent", "Sustained-release agent",
                                     "Extended-release agent", "Modified-release agent",
                                     "Release-modifying agent", "Matrix-forming agent", "Osmotic agent"],
    "solvent":                     ["Solvent", "Cosolvent", "Water-miscible cosolvent"],
    "surfactant":                  ["Surfactant", "Nonionic surfactant", "Anionic surfactant",
                                     "Cationic surfactant", "Wetting agent", "Emulsifying agent",
                                     "Fecal softener", "Detergent"],
    "plasticizer":                 ["Plasticizer"],
    "alkalizing_agent":            ["Alkalizing agent", "Antacid"],
    "acidifying_agent":            ["Acidifying agent", "Acidulant"],
    "humectant":                   ["Humectant", "Emollient"],
    "sweetening_agent":            ["Sweetening agent"],
    "flavoring_agent":             ["Flavoring agent", "Flavor enhancer", "Cooling agent",
                                     "Taste-masking agent"],
    "preservative":                ["Antimicrobial preservative", "Preservative"],
    "antioxidant":                 ["Antioxidant"],
    "buffering_agent":             ["Buffering agent"],
    "suspending_thickening_agent": ["Suspending agent", "Viscosity-increasing agent",
                                     "Thickening agent", "Gelling agent", "Rheology modifier",
                                     "Gel base", "Viscosity-controlling agent"],
    "stabilizing_agent":           ["Stabilizing agent", "Emulsion stabilizer", "Foam stabilizer"],
    "solubilizing_agent":          ["Solubilizing agent", "Dissolution enhancer",
                                     "Solubility enhancing agent", "Complexing agent",
                                     "Chelating agent", "Sequestering agent"],
    "granulation_aid":             ["Granulation aid", "Compression aid"],
    "dispersing_agent":            ["Dispersing agent", "Foaming agent", "Antifoaming agent"],
    "bioadhesive_agent":           ["Bioadhesive material", "Mucoadhesive"],
    "colorant":                    ["Colorant", "Pigment"],
}

ROLE_NAMES = list(ROLE_TAXONOMY.keys())
NUM_ROLES  = len(ROLE_NAMES)

# Build column name → canonical role lookup (lowercase for matching)
COL_TO_CANONICAL = {}
for canonical, hpe_cols in ROLE_TAXONOMY.items():
    for col in hpe_cols:
        COL_TO_CANONICAL[col.lower().strip()] = canonical


def build_unii_to_roles(filled_csv_path: str, unii_lookup_path: str) -> dict:
    """
    Build {unii: set_of_canonical_roles} from the one-hot encoded filled CSV.

    Steps:
      1. Load filled CSV (excipient_names + HPE category binary columns)
      2. Load UNII lookup CSV (excipient_name + unii) — used for name→UNII only
      3. For each excipient row in filled CSV:
           a. Try all name variants to find a UNII
           b. Collect HPE columns where value == 1
           c. Map those columns → canonical roles via ROLE_TAXONOMY
      4. Return {unii: set(canonical_roles)}
    """
    filled_df = pd.read_csv(filled_csv_path)
    unii_df   = pd.read_csv(unii_lookup_path)

    # name → unii lookup (uppercase)
    unii_df['_name_upper'] = unii_df['excipient_name'].str.upper().str.strip()
    name_to_unii = dict(zip(unii_df['_name_upper'], unii_df['unii']))

    # Role columns = everything after the 4 metadata cols
    role_cols = filled_df.columns[4:].tolist()

    unii_to_roles  = {}
    skipped_no_unii = 0

    for _, row in filled_df.iterrows():
        # Try all comma-separated name variants
        name_variants = [n.strip().upper() for n in str(row['excipient_names']).split(',')]
        unii = None
        for name in name_variants:
            if name in name_to_unii:
                unii = name_to_unii[name]
                break

        if not unii:
            skipped_no_unii += 1
            continue

        # Collect active HPE categories
        roles = set()
        for col in role_cols:
            val = row[col]
            if val == 1 or val == 1.0:
                canonical = COL_TO_CANONICAL.get(col.lower().strip())
                if canonical:
                    roles.add(canonical)

        if roles:
            # Union if UNII seen more than once (grade variants, duplicates)
            if unii in unii_to_roles:
                unii_to_roles[unii] |= roles
            else:
                unii_to_roles[unii] = roles

    print(f"  Built UNII→role lookup: {len(unii_to_roles)} excipients with known roles")
    print(f"  Skipped {skipped_no_unii} excipients (no UNII match)")
    return unii_to_roles


def get_role_vector(ingredients_str: str, unii_to_roles: dict) -> np.ndarray:
    """
    Parse inactive_ingredients JSON and return 25-dim binary role vector.
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


def build_features(row) -> np.ndarray:
    """
    Build 23-dim input vector: api_features(20) + dose(1) + route_id(1) + form_id(1)
    """
    api_feats = np.array(row["api_features"],     dtype=np.float32)
    dose      = np.array([row["dose_normalized"]], dtype=np.float32)
    route     = np.array([row["route_id"]],        dtype=np.float32)
    form      = np.array([row["form_id"]],          dtype=np.float32)
    return np.concatenate([api_feats, dose, route, form])


def build_split(df_split, unii_to_roles: dict):
    X, y = [], []
    skipped = 0
    for _, row in df_split.iterrows():
        target = get_role_vector(row["inactive_ingredients"], unii_to_roles)
        if target.sum() == 0:
            skipped += 1
            continue
        X.append(build_features(row))
        y.append(target)
    print(f"    Built {len(X)} samples, skipped {skipped} (no roles mapped)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    import os
    os.makedirs("role_predictor", exist_ok=True)

    print("=" * 55)
    print("Building Role Prediction Dataset (v2 — one-hot source)")
    print("=" * 55)

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

    print("\n[3] Building UNII → role lookup from one-hot filled CSV...")
    unii_to_roles = build_unii_to_roles(FILLED_CSV_PATH, UNII_LOOKUP_PATH)

    print("\n[4] Building train split...")
    X_train, y_train = build_split(train_df, unii_to_roles)

    print("\n[5] Building val split...")
    X_val, y_val = build_split(val_df, unii_to_roles)

    print("\n[6] Building test split...")
    X_test, y_test = build_split(test_df, unii_to_roles)

    role_counts = {
        ROLE_NAMES[i]: int(y_train[:, i].sum())
        for i in range(NUM_ROLES)
    }

    print(f"\n[7] Role distribution in train set:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = count / len(y_train) * 100
        print(f"    {role:<30} {count:>4} / {len(y_train)} ({pct:.1f}%)")

    dataset = {
        "X_train":     X_train,
        "y_train":     y_train,
        "X_val":       X_val,
        "y_val":       y_val,
        "X_test":      X_test,
        "y_test":      y_test,
        "role_names":  ROLE_NAMES,
        "role_counts": role_counts,
        "num_roles":   NUM_ROLES,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\n{'=' * 55}")
    print(f"[OK] Dataset saved to: {OUTPUT_PATH}")
    print(f"     Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"     Roles: {NUM_ROLES} → {ROLE_NAMES}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()