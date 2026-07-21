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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config import ROLE_NAMES, MODEL_PATH, NUM_ROLES
from dataset import build_dataset


def drop_rare_classes(X, y, w, min_count=2):
    """Stratified split needs >=2 examples per class; drop rarer ones."""
    counts = pd.Series(y).value_counts()
    rare = counts[counts < min_count].index.tolist()
    if rare:
        rare_names = [ROLE_NAMES[c] for c in rare]
        print(f"Dropping {len(rare)} classes with <{min_count} examples: {rare_names}")
        keep = ~np.isin(y, rare)
        X, y, w = X[keep], y[keep], w[keep]
    return X, y, w


def train_mlp(X_train, y_train, sample_weight=None):
    # sklearn's MLPClassifier doesn't support sample_weight directly;
    # approximate weighting by oversampling high-weight rows when pass2 is included.
    if sample_weight is not None and not np.allclose(sample_weight, 1.0):
        reps = np.clip((sample_weight * 3).astype(int), 1, 3)
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

    X, y, w, unii_to_roles, formulation_excipients = build_dataset(
        include_pass2=args.include_pass2,
        pass2_confidence_threshold=args.pass2_threshold,
    )
    X, y, w = drop_rare_classes(X, y, w)

    # XGBoost requires contiguous 0..n_classes-1 labels; dropping rare
    # classes leaves gaps in the original ROLE_TO_IDX indices, so re-encode.
    # (harmless for MLP too, we just invert it consistently at inference.)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y_encoded, w, test_size=0.2, random_state=42, stratify=y_encoded
    )

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
            "X_val": X_val, "y_val": y_val,  # y_val is ENCODED; evaluate.py decodes via label_encoder
        }, f)
    print(f"[OK] Model saved to {MODEL_PATH}")
    print("Run evaluate.py next to check performance.")


if __name__ == "__main__":
    main()