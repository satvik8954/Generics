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
    df = pd.read_csv("Data/f3.csv")
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

    output_path = "Data/api_features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"\n[OK] Saved {len(features_df)} API feature vectors to {output_path}")
    print(f"     Columns: {list(features_df.columns)}")


if __name__ == "__main__":
    main()
