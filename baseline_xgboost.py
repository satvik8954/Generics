"""
baseline_xgboost.py — Small XGBoost baseline for excipient prediction.

Trains one-vs-rest binary classifiers on simple features and evaluates
precision/recall/F1/Jaccard at K on the test split.

Usage:
  python baseline_xgboost.py
  python baseline_xgboost.py --max-labels 100 --sample 0.2
"""

import argparse
import pickle
import time
import numpy as np
import torch

try:
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError
except Exception as exc:
    raise SystemExit(
        "xgboost is required for this baseline. Install with: pip install xgboost"
    ) from exc

try:
    from sklearn.preprocessing import MultiLabelBinarizer
except Exception as exc:
    raise SystemExit(
        "scikit-learn is required for this baseline. Install with: pip install scikit-learn"
    ) from exc

from metrics import precision_at_k, recall_at_k, f1_at_k, jaccard_at_k

try:
    from tqdm import tqdm
except Exception as exc:
    raise SystemExit(
        "tqdm is required for progress bars. Install with: pip install tqdm"
    ) from exc


def build_feature_matrix(df):
    api_feats = np.vstack(df["api_features"].to_numpy())
    dose = df["dose_normalized"].to_numpy().reshape(-1, 1)
    per_unit = df["per_unit_id"].to_numpy().reshape(-1, 1)
    route = df["route_id"].to_numpy().reshape(-1, 1)
    form = df["form_id"].to_numpy().reshape(-1, 1)

    return np.hstack([api_feats, dose, per_unit, route, form]).astype(np.float32)


def sample_rows(df, fraction, seed):
    if fraction >= 1.0:
        return df
    return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="XGBoost baseline")
    parser.add_argument("--max-labels", type=int, default=0, help="Limit excipient labels (0 = all)")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of train/val/test rows to use")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for OVR/XGBoost")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if XGBoost has CUDA support")
    args = parser.parse_args()

    with open("split_data.pkl", "rb") as f:
        split_data = pickle.load(f)

    train_df = sample_rows(split_data["train_df"], args.sample, seed=42)
    val_df = sample_rows(split_data["val_df"], args.sample, seed=42)
    test_df = sample_rows(split_data["test_df"], args.sample, seed=42)

    X_train = build_feature_matrix(train_df)
    X_val = build_feature_matrix(val_df)
    X_test = build_feature_matrix(test_df)

    y_train_raw = train_df["excipient_ids"].tolist()
    y_val_raw = val_df["excipient_ids"].tolist()
    y_test_raw = test_df["excipient_ids"].tolist()

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_raw)
    y_val = mlb.transform(y_val_raw)
    y_test = mlb.transform(y_test_raw)

    if args.max_labels and args.max_labels < y_train.shape[1]:
        y_train = y_train[:, : args.max_labels]
        y_val = y_val[:, : args.max_labels]
        y_test = y_test[:, : args.max_labels]

    label_count = y_train.shape[1]
    if label_count > 500 and args.max_labels == 0:
        print(
            f"Warning: training {label_count} one-vs-rest models can take a long time. "
            "Consider --max-labels or --sample to speed up."
        )

    tree_method = "gpu_hist" if args.gpu else "hist"

    print(
        "Training XGBoost baseline... "
        f"samples={X_train.shape[0]}, features={X_train.shape[1]}, labels={label_count}"
    )
    start = time.time()
    models = []
    use_gpu = args.gpu
    for idx in tqdm(range(label_count), desc="Training labels", unit="label"):
        params = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist" if use_gpu else "hist",
            n_jobs=args.n_jobs,
        )
        clf = XGBClassifier(**params)
        try:
            clf.fit(X_train, y_train[:, idx])
        except XGBoostError:
            if use_gpu:
                print("GPU training failed; retrying remaining labels on CPU.")
                use_gpu = False
                clf = XGBClassifier(**{**params, "tree_method": "hist"})
                clf.fit(X_train, y_train[:, idx])
            else:
                raise
        models.append(clf)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    print("Scoring...")
    y_scores = np.zeros((X_test.shape[0], label_count), dtype=np.float32)
    for idx, model in enumerate(tqdm(models, desc="Scoring labels", unit="label")):
        probs = model.predict_proba(X_test)
        y_scores[:, idx] = probs[:, 1]

    y_scores_t = torch.tensor(y_scores, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    print("\nEVALUATION RESULTS")
    print("Metric".ljust(12) + "".join([f"@{k}".rjust(10) for k in args.k]))
    print("-" * (12 + 10 * len(args.k)))

    for metric_name, metric_fn in [
        ("Precision", precision_at_k),
        ("Recall", recall_at_k),
        ("F1", f1_at_k),
        ("Jaccard", jaccard_at_k),
    ]:
        row = metric_name.ljust(12)
        for k in args.k:
            row += f"{metric_fn(y_scores_t, y_test_t, k):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
