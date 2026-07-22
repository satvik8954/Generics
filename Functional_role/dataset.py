"""
dataset.py — turns weak_labels_high_confidence.csv (and optionally
weak_labels_pass2.csv) into (X, y) arrays ready for training.
"""

import numpy as np
import pandas as pd

from config import ROLE_TO_IDX, HIGH_CONF_CSV_PATH, PASS2_CSV_PATH
from roles import build_unii_to_roles
from features import build_feature_vector, load_formulation_excipients, load_api_features


def build_dataset(include_pass2: bool = False, pass2_confidence_threshold: float = 0.0,
                   use_api_features: bool = True):
    """
    Returns (X, y, sample_weight, groups, unii_to_roles, formulation_excipients)

    groups[i] = the row_idx (formulation id) that example i came from. Use
    this with a grouped split (e.g. StratifiedGroupKFold) instead of a plain
    train_test_split, so that two excipients from the SAME formulation never
    get split across train/val — otherwise co-occurrence features let val
    rows leak information about formulations the model already saw in train.

    include_pass2: if True, also include pass2 rows (filtered by
        pass2_confidence_threshold) with sample_weight = confidence.
        pass1 rows always get sample_weight = 1.0.
    use_api_features: if True, appends the 20 z-score-normalized RDKit
        descriptors per API (see config.API_FEATURES_CSV).
    """
    print("Loading excipient capability map...")
    unii_to_roles = build_unii_to_roles()

    print("Loading formulations for co-occurrence context...")
    formulation_excipients = load_formulation_excipients()

    api_to_feats = None
    if use_api_features:
        print("Loading API descriptors...")
        api_to_feats = load_api_features()

    print("Loading pass1 (high-confidence) labels...")
    df = pd.read_csv(HIGH_CONF_CSV_PATH)
    df["sample_weight"] = 1.0

    if include_pass2:
        print(f"Loading pass2 labels (confidence >= {pass2_confidence_threshold})...")
        df2 = pd.read_csv(PASS2_CSV_PATH)
        df2 = df2[df2["confidence"] >= pass2_confidence_threshold].copy()
        df2["sample_weight"] = df2["confidence"]
        print(f"  {len(df2)} pass2 rows included")
        df = pd.concat([df, df2[df.columns]], ignore_index=True)

    print(f"Building feature matrix for {len(df)} examples...")
    X, y, w, groups = [], [], [], []
    for _, row in df.iterrows():
        other_uniis = formulation_excipients.get(row["row_idx"], [])
        feats = build_feature_vector(row["excipient_unii"], row["dosage_form"],
                                      other_uniis, unii_to_roles,
                                      api_unii=row["api_unii"], api_to_feats=api_to_feats)
        X.append(feats)
        y.append(ROLE_TO_IDX[row["role"]])
        w.append(row["sample_weight"])
        groups.append(row["row_idx"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    w = np.array(w, dtype=np.float32)
    groups = np.array(groups)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y, w, groups, unii_to_roles, formulation_excipients