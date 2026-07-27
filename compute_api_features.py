"""
compute_api_features.py — computes the same 20 RDKit descriptors used in
data/api_features.csv (see config.API_FEATURE_COLS) for new APIs, and
appends them to that file.

Usage:
    python compute_api_features.py

Edit NEW_APIS below to add/remove entries. Each tuple is (api_unii, smiles).
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

API_FEATURES_CSV = "Data/api_features.csv"  # same path as config.API_FEATURES_CSV

# column order MUST match config.API_FEATURE_COLS exactly, or the model
# will read the wrong descriptor into the wrong feature slot
API_FEATURE_COLS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings", "RingCount",
    "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons", "MolMR",
    "LabuteASA", "BalabanJ", "BertzCT", "HallKierAlpha", "NumSaturatedRings",
    "NumHeteroatoms", "NHOHCount",
]

# (api_unii, smiles) — the three APIs missing from api_features.csv,
# UNIIs verified against FDA GSRS/precision.fda.gov
NEW_APIS = [
    ("XZ7BG04GJX", "O=C2N1/C(=C(/C=C)CS[C@@H]1[C@@H]2NC(=O)C(=N\\OCC(=O)O)/c3nc(sc3)N)C(=O)O"),  # cefixime anhydrous
    ("7355X3ROTS", "CN1CC[C@@]23CCCC[C@@H]2[C@@H]1Cc4c3cc(cc4)OC"),  # dextromethorphan base
    ("CC995ZMV90", "Cl.NC1=C(Br)C=C(Br)C=C1CN[C@H]1CC[C@H](O)CC1"),  # ambroxol hydrochloride
]


def compute_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "MolMR": Descriptors.MolMR(mol),
        "LabuteASA": Descriptors.LabuteASA(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "NHOHCount": Descriptors.NHOHCount(mol),
    }


def main():
    rows = []
    for unii, smiles in NEW_APIS:
        print(f"Computing descriptors for {unii} ({smiles})...")
        feats = compute_descriptors(smiles)
        row = {"api_unii": unii, "smiles": smiles, **feats}
        rows.append(row)

    new_df = pd.DataFrame(rows, columns=["api_unii", "smiles"] + API_FEATURE_COLS)

    if os.path.exists(API_FEATURES_CSV):
        existing = pd.read_csv(API_FEATURES_CSV)
        dupes = existing[existing["api_unii"].isin(new_df["api_unii"])]
        if len(dupes):
            print(f"[WARN] {len(dupes)} of the new UNIIs already exist in "
                  f"{API_FEATURES_CSV} — dropping existing rows for those "
                  f"UNIIs and replacing with the freshly computed ones:")
            print(dupes["api_unii"].tolist())
            existing = existing[~existing["api_unii"].isin(new_df["api_unii"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        print(f"[WARN] {API_FEATURES_CSV} not found — creating a new file "
              f"with just these {len(new_df)} rows.")
        combined = new_df

    combined.to_csv(API_FEATURES_CSV, index=False)
    print(f"\n[OK] {API_FEATURES_CSV} now has {len(combined)} total APIs "
          f"({len(new_df)} newly added/updated).")
    print(new_df.to_string(index=False))


if __name__ == "__main__":
    main()
