"""
evaluate.py — Evaluation script for ExciPick Role Predictor

Prints per-role F1, precision, recall + macro averages.

Usage:
    python role_predictor/evaluate.py
"""

import pickle
import torch
import numpy as np
from model import RolePredictor
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "role_predictor/role_dataset.pkl"
MODEL_PATH   = "role_predictor/best_role_model.pt"
INPUT_DIM    = 23
HIDDEN_DIM   = 128
DROPOUT      = 0.3
THRESHOLD    = 0.5
BATCH_SIZE   = 512


def evaluate(model, loader, device, role_names, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = torch.sigmoid(model(X_batch)).cpu()
            all_preds.append(preds)
            all_targets.append(y_batch)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    pred_bin = (preds >= threshold).float()

    num_roles = len(role_names)
    results = []

    for i in range(num_roles):
        tp = (pred_bin[:, i] * targets[:, i]).sum().item()
        fp = (pred_bin[:, i] * (1 - targets[:, i])).sum().item()
        fn = ((1 - pred_bin[:, i]) * targets[:, i]).sum().item()
        tn = ((1 - pred_bin[:, i]) * (1 - targets[:, i])).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        support   = int(targets[:, i].sum().item())

        results.append({
            "role":      role_names[i],
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   support,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })

    return results, preds, targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    X_test     = torch.tensor(dataset["X_test"], dtype=torch.float32)
    y_test     = torch.tensor(dataset["y_test"], dtype=torch.float32)
    role_names = dataset["role_names"]
    num_roles  = dataset["num_roles"]

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Load model
    model = RolePredictor(
        input_dim=INPUT_DIM,
        num_roles=num_roles,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")

    # Evaluate
    results, preds, targets = evaluate(
        model, test_loader, device, role_names, threshold=THRESHOLD
    )

    # Print report
    print(f"\n{'='*65}")
    print(f"ROLE PREDICTION EVALUATION (threshold={THRESHOLD})")
    print(f"Test set: {len(X_test)} formulations")
    print(f"{'='*65}")
    print(f"{'Role':<25} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>8}")
    print(f"{'-'*65}")

    f1_scores = []
    for r in sorted(results, key=lambda x: -x["f1"]):
        print(
            f"{r['role']:<25} "
            f"{r['precision']:>9.3f} "
            f"{r['recall']:>9.3f} "
            f"{r['f1']:>9.3f} "
            f"{r['support']:>8}"
        )
        f1_scores.append(r["f1"])

    macro_f1 = np.mean(f1_scores)
    print(f"{'-'*65}")
    print(f"{'MACRO AVERAGE':<25} {'':>9} {'':>9} {macro_f1:>9.3f}")
    print(f"{'='*65}")

    # Threshold sweep — find optimal threshold
    print(f"\nThreshold sweep (macro F1):")
    print(f"{'Threshold':>10} {'Macro F1':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_bin = (preds >= t).float()
        tp = (pred_bin * targets).sum(dim=0)
        fp = (pred_bin * (1 - targets)).sum(dim=0)
        fn = ((1 - pred_bin) * targets).sum(dim=0)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()
        print(f"{t:>10.1f} {f1:>10.4f}")

    if macro_f1 >= 0.70:
        print(f"\n✓ SUCCESS — Macro F1 {macro_f1:.3f} >= 0.70 threshold")
        print(f"  Hierarchical model approach is validated.")
    else:
        print(f"\n✗ Below target — Macro F1 {macro_f1:.3f} < 0.70")
        print(f"  Review low-F1 roles above.")


if __name__ == "__main__":
    main()
