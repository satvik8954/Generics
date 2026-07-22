"""
train.py — trains a Task B role-assignment classifier on weak labels.

Usage:
    python train.py                 # MLP, pass1 only (default, safest baseline)
    python train.py --model xgboost # XGBoost instead of MLP
    python train.py --include-pass2 --pass2-threshold 0.2   # add pass2 rows

Saves the trained model + role names to config.MODEL_PATH.
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config import ROLE_NAMES, MODEL_PATH, NUM_ROLES
from dataset import build_dataset
from features import fit_coocc_scaler, apply_coocc_scaler


def drop_rare_classes(X, y, w, groups, min_count=2):
    """Stratified split needs >=2 examples per class; drop rarer ones."""
    counts = pd.Series(y).value_counts()
    rare = counts[counts < min_count].index.tolist()
    if rare:
        rare_names = [ROLE_NAMES[c] for c in rare]
        print(f"Dropping {len(rare)} classes with <{min_count} examples: {rare_names}")
        keep = ~np.isin(y, rare)
        X, y, w, groups = X[keep], y[keep], w[keep], groups[keep]
    return X, y, w, groups


def train_mlp(X_train, y_train, sample_weight=None):
    # sklearn's MLPClassifier doesn't support sample_weight directly;
    # approximate weighting by oversampling high-weight rows.
    #
    # Old formula was `np.clip((sample_weight * 3).astype(int), 1, 3)`, which
    # TRUNCATES rather than rounds: since real pass2 confidences have median
    # ~0.0125 and 75th percentile ~0.29, over 94% of pass2 rows truncate to
    # reps=1 regardless of their actual confidence — a 0.02-confidence row
    # and a 0.6-confidence row got treated identically. This version rounds
    # (not truncates) and spreads across a wider 1-5 range so confidence
    # differences actually show up in how often a row gets duplicated:
    #   conf in [0.00,0.25) -> 1x   conf in [0.50,0.75) -> 3x
    #   conf in [0.25,0.50) -> 2x   conf in [0.75,1.00]  -> 4-5x
    if sample_weight is not None and not np.allclose(sample_weight, 1.0):
        reps = np.clip(1 + np.floor(sample_weight * 4).astype(int), 1, 5)
        X_train = np.repeat(X_train, reps, axis=0)
        y_train = np.repeat(y_train, reps, axis=0)
        print(f"  Oversampled by confidence -> {len(X_train)} effective training rows")

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                         alpha=1e-4, max_iter=300, early_stopping=True, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train, y_train, sample_weight=None):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=NUM_ROLES,
        eval_metric="mlogloss", random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "xgboost"], default="mlp")
    parser.add_argument("--include-pass2", action="store_true")
    parser.add_argument("--pass2-threshold", type=float, default=0.2)
    args = parser.parse_args()

    X, y, w, groups, unii_to_roles, formulation_excipients = build_dataset(
        include_pass2=args.include_pass2,
        pass2_confidence_threshold=args.pass2_threshold,
    )
    X, y, w, groups = drop_rare_classes(X, y, w, groups)

    # XGBoost requires contiguous 0..n_classes-1 labels; dropping rare
    # classes leaves gaps in the original ROLE_TO_IDX indices, so re-encode.
    # (harmless for MLP too, we just invert it consistently at inference.)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Grouped + stratified split: excipients from the SAME formulation
    # (same row_idx / group) always land entirely in train OR entirely in
    # val, never split across both. Plain train_test_split doesn't know
    # about groups, so two excipients from formulation #17 could end up on
    # opposite sides of the split — and since co-occurrence features for one
    # are derived from the other, that's real information leakage between
    # "train" and "val", inflating validation accuracy. StratifiedGroupKFold
    # keeps groups intact while still trying to balance classes across folds.
    n_splits = 5  # -> ~20% val, matching the old test_size=0.2
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(X, y_encoded, groups=groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    w_train, w_val = w[train_idx], w[val_idx]
    print(f"Grouped split: {len(set(groups[train_idx]))} train formulations, "
          f"{len(set(groups[val_idx]))} val formulations "
          f"(0 overlap: {len(set(groups[train_idx]) & set(groups[val_idx]))})")

    # Fit the co-occurrence-block scaler on TRAIN ONLY, then apply the same
    # fitted scaler to val (never fit on val — that would leak val stats
    # into the scaling). Saved into the model file so evaluate.py/predict.py
    # apply the identical transform at eval/inference time.
    coocc_scaler = fit_coocc_scaler(X_train)
    X_train = apply_coocc_scaler(X_train, coocc_scaler)
    X_val = apply_coocc_scaler(X_val, coocc_scaler)

    print(f"\nTraining {args.model}...")
    if args.model == "mlp":
        clf = train_mlp(X_train, y_train, w_train)
    else:
        clf = train_xgboost(X_train, y_train, w_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": clf, "model_type": args.model,
            "role_names": ROLE_NAMES,
            "label_encoder": label_encoder,  # maps encoded class -> original ROLE_TO_IDX index
            "coocc_scaler": coocc_scaler,    # apply via features.apply_coocc_scaler before predict
            "X_val": X_val, "y_val": y_val,  # already scaled; y_val is ENCODED, decode via label_encoder
        }, f)
    print(f"[OK] Model saved to {MODEL_PATH}")
    print("Run evaluate.py next to check performance.")


if __name__ == "__main__":
    main()