"""
preprocess.py -- Prepares Data/mapped_formulations.csv for the ExciPick model.

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
import re
from collections import Counter

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
INPUT_PATH = "Data/final.csv"
FEATURES_PATH = "Data/api_features.csv"
OUTPUT_PATH = "processed_data.pkl"

MIN_EXCIPIENTS = 2
MAX_EXCIPIENTS = 30
# EXCIPIENT_MIN_FREQ = 10
# TARGET_FORMS = [
#     "TABLET", "TABLET, FILM COATED", "TABLET, EXTENDED RELEASE", 
#     "TABLET, DELAYED RELEASE", "TABLET, FILM COATED, EXTENDED RELEASE"
# ]
# TARGET_FORMS = ["TABLET"]
TARGET_FORMS = [
    "TABLET",
    "TABLET, FILM COATED", 
    "TABLET, EXTENDED RELEASE",
    "TABLET, DELAYED RELEASE",
    "TABLET, FILM COATED, EXTENDED RELEASE",
    "TABLET, COATED",
    "CAPSULE",  # add capsules back
]
EXCIPIENT_MIN_FREQ = 5

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons",
    "MolMR", "LabuteASA", "BalabanJ", "BertzCT", "HallKierAlpha",
    "NumSaturatedRings", "NumHeteroatoms", "NHOHCount",
]


def normalize_excipient_name(name):
    if not isinstance(name, str):
        return ""

    cleaned = name.upper().strip()

    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) == 2:
            cleaned = f"{parts[1]} {parts[0]}"

    cleaned = re.sub(r"\s*\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\s+[K]?\d+(\.\d+)?\s*$", "", cleaned)
    cleaned = re.sub(r",?\s*(UNSPECIFIED|TYPE [A-Z])$", "", cleaned)

    return cleaned.strip()


def parse_excipient_names(json_str):
    """Extract excipient names from inactive_ingredients JSON string."""
    try:
        ingredients = json.loads(json_str)
        names = list(
            set(
                normalize_excipient_name(item.get("name", ""))
                for item in ingredients
            )
        )
        return [name for name in names if name]
    except Exception:
        return []


def extract_smiles(json_str):
    """Extract the first SMILES string from api_smiles_map JSON."""
    try:
        smap = json.loads(json_str)
        return list(smap.values())[0]
    except Exception:
        return None

def get_exc_unii_set(inactive_str):
    """Extract a sorted tuple of excipient UNIIs from JSON."""
    try:
        items = json.loads(inactive_str)
        return tuple(sorted(set(i.get('unii', '') for i in items if 'unii' in i)))
    except Exception:
        return tuple()


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
    
    # Precise Deduplication
    print(f"[0] Rows before dedup: {len(df)}")
    df['exc_unii_set'] = df['inactive_ingredients'].apply(get_exc_unii_set)
    df = df.drop_duplicates(subset=['api_unii', 'dose_mg', 'exc_unii_set', 'route', 'primary_dosage_form'])
    print(f"    Rows after dedup: {len(df)}")
    
    # Filter out missing route or form
    df = df.dropna(subset=['route', 'primary_dosage_form'])
    print(f"    Valid formulations (all routes/forms): {len(df)}")
    
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
    df = df[df["primary_dosage_form"].isin(TARGET_FORMS)]
    print(f"    After form filter: {len(df)} rows")

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