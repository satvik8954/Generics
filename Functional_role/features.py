"""
features.py — builds the model input feature vector for one
(excipient, formulation) example. Shared by train.py, evaluate.py, predict.py
so train/inference features are always constructed identically.

Feature layout (74-dim with the current 24-role taxonomy + 20 API descriptors):
    [0:24]  target excipient's RAW (unfiltered) HPE capability mask
    [24:30] dosage-form bucket features (is_tablet, is_capsule, is_liquid,
            is_extended_release, is_coated, is_chewable_odt)
    [30:54] co-occurrence context: count of OTHER excipients in the same
            formulation whose raw capability includes each role
    [54:74] API physicochemical descriptors (RDKit, z-score normalized) —
            MolWt, LogP, TPSA, etc. Same API -> same 20 values regardless
            of dosage form/excipients, but lets the model learn e.g.
            "highly hygroscopic/acid-labile APIs correlate with certain
            excipient role patterns" if such patterns exist in the data.
"""

import json
import numpy as np
import pandas as pd

from config import (NUM_ROLES, ROLE_TO_IDX, ORAL_CSV,
                     API_FEATURES_CSV, API_FEATURE_COLS, NUM_API_FEATURES)
from roles import dosage_form_bucket_features


def raw_capability_mask(unii, unii_to_roles) -> np.ndarray:
    mask = np.zeros(NUM_ROLES, dtype=np.float32)
    for role in unii_to_roles.get(unii, set()):
        mask[ROLE_TO_IDX[role]] = 1.0
    return mask


def load_formulation_excipients(oral_csv_path: str = ORAL_CSV) -> dict:
    """row_idx -> list of excipient uniis, used for co-occurrence features."""
    oral = pd.read_csv(oral_csv_path)
    formulation_excipients = {}
    for idx, row in oral.iterrows():
        try:
            items = json.loads(row["inactive_ingredients"])
            formulation_excipients[idx] = [str(it.get("unii", "")).strip()
                                            for it in items if it.get("unii")]
        except Exception:
            continue
    return formulation_excipients


def load_api_features(path: str = API_FEATURES_CSV) -> dict:
    """
    Returns {api_unii: np.array(NUM_API_FEATURES,)} with columns z-score
    normalized (mean 0, std 1) using stats computed across this file, since
    RDKit descriptors have wildly different native scales (MolWt ~100-500
    vs FractionCSP3 ~0-1 vs BertzCT in the hundreds/thousands) and an MLP
    is sensitive to that.
    """
    df = pd.read_csv(path)
    means = df[API_FEATURE_COLS].mean()
    stds = df[API_FEATURE_COLS].std().replace(0, 1.0)  # avoid div-by-zero on constant cols

    api_to_feats = {}
    for _, row in df.iterrows():
        vals = (row[API_FEATURE_COLS] - means) / stds
        api_to_feats[row["api_unii"]] = vals.values.astype(np.float32)

    print(f"  Loaded API descriptors for {len(api_to_feats)} APIs")
    return api_to_feats


def build_feature_vector(target_unii: str, dosage_form: str, other_uniis: list,
                          unii_to_roles: dict, api_unii: str = None,
                          api_to_feats: dict = None) -> np.ndarray:
    """
    other_uniis: all excipient uniis in the same formulation EXCLUDING target_unii
    api_unii / api_to_feats: optional — if both given, appends API descriptors.
        If api_unii isn't found in api_to_feats, falls back to zeros (mean,
        since features are z-score normalized) rather than dropping the row.
    """
    target_mask = raw_capability_mask(target_unii, unii_to_roles)
    form_feats = dosage_form_bucket_features(dosage_form)

    co_counts = np.zeros(NUM_ROLES, dtype=np.float32)
    for other_unii in other_uniis:
        if other_unii == target_unii:
            continue
        for role in unii_to_roles.get(other_unii, set()):
            co_counts[ROLE_TO_IDX[role]] += 1.0

    parts = [target_mask, form_feats, co_counts]

    if api_to_feats is not None:
        api_feats = api_to_feats.get(api_unii)
        if api_feats is None:
            api_feats = np.zeros(NUM_API_FEATURES, dtype=np.float32)
        parts.append(api_feats)

    return np.concatenate(parts)


FEATURE_DIM = NUM_ROLES + 6 + NUM_ROLES  # + NUM_API_FEATURES if api_to_feats is used