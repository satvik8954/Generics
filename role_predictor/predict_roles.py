"""
predict_roles.py — Inference script for ExciPick Role Predictor

Usage:
    python role_predictor/predict_roles.py --api U3H27498KS --dose 25 --unit 1 --route ORAL --form TABLET
    python role_predictor/predict_roles.py --batch-from-oral 5
"""

import argparse
import pickle
import csv
import json
import torch
import numpy as np
from model import RolePredictor
from build_role_dataset import build_unii_to_roles, ROLES_CSV_PATH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH   = "role_predictor/role_dataset.pkl"
MODEL_PATH     = "role_predictor/best_role_model.pt"
PROCESSED_PATH = "processed_data.pkl"
ORAL_ONLY_PATH = "Data/oral_only.csv"
INPUT_DIM      = 23
HIDDEN_DIM     = 128
DROPOUT        = 0.3
THRESHOLD      = 0.5

ROLE_DESCRIPTIONS = {
    "binder":             "Holds tablet together during compression",
    "filler":             "Adds bulk to the tablet (diluent)",
    "disintegrant":       "Helps tablet break apart on ingestion",
    "lubricant":          "Prevents sticking during manufacturing",
    "glidant":            "Improves powder flow",
    "coating_agent":      "Protects core or controls release",
    "colorant":           "Provides color/appearance",
    "controlled_release": "Modulates drug release rate",
    "solvent":            "Dissolves components during processing",
    "surfactant":         "Improves wetting and dissolution",
    "plasticizer":        "Adds flexibility to coatings",
    "alkalizing_agent":   "Adjusts pH of the formulation",
    "humectant":          "Retains moisture",
}


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_everything():
    with open(PROCESSED_PATH, "rb") as f:
        data = pickle.load(f)
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    role_names = dataset["role_names"]
    num_roles  = dataset["num_roles"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RolePredictor(
        input_dim=INPUT_DIM,
        num_roles=num_roles,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    unii_to_roles = build_unii_to_roles(ROLES_CSV_PATH)
    return model, data, role_names, device, unii_to_roles


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict_roles(model, data, role_names, device, api_unii, dose_mg, unit, route, form):
    df             = data["df"]
    route_vocab    = data["route_vocab"]
    form_vocab     = data["form_vocab"]
    per_unit_vocab = data["per_unit_vocab"]
    dose_stats     = data["dose_stats"]

    api_rows = df[df["api_unii"] == api_unii]
    if api_rows.empty:
        raise ValueError(f"API UNII '{api_unii}' not found.")

    api_features = np.array(api_rows.iloc[0]["api_features"], dtype=np.float32)
    log_dose     = np.log10(max(dose_mg, 1e-9))
    dose_norm    = (log_dose - dose_stats["mean"]) / dose_stats["std"]

    for name, vocab, val in [
        ("unit",  per_unit_vocab, unit),
        ("route", route_vocab,    route),
        ("form",  form_vocab,     form),
    ]:
        if val not in vocab:
            raise ValueError(f"{name} '{val}' not in vocab. Valid: {list(vocab.keys())[:8]}")

    features = np.concatenate([
        api_features,
        [dose_norm],
        [route_vocab[route]],
        [form_vocab[form]],
    ]).astype(np.float32)

    x = torch.tensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()

    results = [(role_names[i], float(probs[i])) for i in range(len(role_names))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_ground_truth_roles(data, role_names, unii_to_roles, api_unii, dose_mg, unit, route, form, dose_tol=10.0):
    df = data["df"]
    candidates = df[
        (df["api_unii"] == api_unii) &
        (df["route"] == route) &
        (df["primary_dosage_form"] == form) &
        (df["denominator_unit"] == unit)
    ].copy()

    if candidates.empty:
        return None, "No matching formulation found"

    candidates["dose_diff"] = (candidates["dose_mg"] - dose_mg).abs()
    candidates = candidates[candidates["dose_diff"] <= dose_tol]

    if candidates.empty:
        return None, f"No dose match within ±{dose_tol}mg"

    row = candidates.sort_values("dose_diff").iloc[0]

    gt_roles = set()
    try:
        items = json.loads(row["inactive_ingredients"])
        for item in items:
            unii = str(item.get("unii", "")).strip()
            for r in unii_to_roles.get(unii, set()):
                if r in role_names:
                    gt_roles.add(r)
    except Exception:
        pass

    return gt_roles, None


# ─────────────────────────────────────────────
# FORMAT
# ─────────────────────────────────────────────

def format_output(results, threshold, gt_roles=None):
    W = 68
    lines = [
        f"\n{'─'*W}",
        "  PREDICTED FUNCTIONAL ROLES",
        f"{'─'*W}",
        f"  {'Role':<22} {'Prob':>6}  {'Status':<10} {'Match':>6}  Description",
        f"{'─'*W}",
    ]

    predicted, tp, fp, fn = [], 0, 0, 0

    for role, prob in results:
        is_pred = prob >= threshold
        marker  = "→" if is_pred else " "
        status  = "✓ NEEDED" if is_pred else "✗ skip"
        desc    = ROLE_DESCRIPTIONS.get(role, "")

        if gt_roles is not None:
            in_gt = role in gt_roles
            if is_pred and in_gt:        match = "✓ TP"; tp += 1
            elif is_pred and not in_gt:  match = "✗ FP"; fp += 1
            elif not is_pred and in_gt:  match = "✗ FN"; fn += 1
            else:                        match = "  TN"
        else:
            match = ""

        lines.append(f"  {marker} {role:<21} {prob:>5.1%}  {status:<10} {match:<6}  {desc}")
        if is_pred:
            predicted.append(role)

    lines.append(f"{'─'*W}")
    lines.append(f"  Predicted ({len(predicted)}): {', '.join(predicted) or 'none'}")

    if gt_roles is not None:
        lines.append(f"  Ground truth ({len(gt_roles)}): {', '.join(sorted(gt_roles)) or 'none'}")
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        lines.append(f"\n  Matched {tp}/{len(gt_roles)} GT roles  |  "
                     f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    lines.append(f"{'─'*W}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_oral_rows(limit):
    rows = []
    try:
        with open(ORAL_ONLY_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if len(rows) >= limit:
                    break
                try:
                    dose_val = float(row.get("dose_mg", ""))
                except ValueError:
                    dose_val = None
                rows.append({
                    "api":   row.get("api_unii"),
                    "dose":  dose_val,
                    "unit":  row.get("denominator_unit"),
                    "route": row.get("route"),
                    "form":  row.get("primary_dosage_form"),
                })
    except FileNotFoundError:
        print(f"Error: {ORAL_ONLY_PATH} not found.")
    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick Role Predictor")
    parser.add_argument("--api",             type=str,   default=None)
    parser.add_argument("--dose",            type=float, default=None)
    parser.add_argument("--unit",            type=str,   default=None)
    parser.add_argument("--route",           type=str,   default=None)
    parser.add_argument("--form",            type=str,   default=None)
    parser.add_argument("--threshold",       type=float, default=THRESHOLD)
    parser.add_argument("--dose-tol",        type=float, default=10.0)
    parser.add_argument("--batch-from-oral", type=int,   default=0)
    args = parser.parse_args()

    print("Loading model and data...")
    model, data, role_names, device, unii_to_roles = load_everything()
    print(f"Using device: {device}\n")

    # Build input rows
    if args.batch_from_oral > 0:
        input_rows = load_oral_rows(args.batch_from_oral)
        if not input_rows:
            exit(1)
    else:
        input_rows = [{
            "api":   args.api,
            "dose":  args.dose,
            "unit":  args.unit,
            "route": args.route,
            "form":  args.form,
        }]

    batch_results = []

    for idx, row in enumerate(input_rows, 1):
        print(f"{'='*68}")
        label = "INPUT" if len(input_rows) == 1 else f"INPUT ({idx}/{len(input_rows)})"
        print(f"  {label}")
        print(f"  API: {row['api']}  |  Dose: {row['dose']}mg  |  "
              f"Unit: {row['unit']}  |  Form: {row['form']}")
        print(f"{'='*68}")

        try:
            results = predict_roles(
                model, data, role_names, device,
                api_unii=row["api"],
                dose_mg=row["dose"],
                unit=row["unit"],
                route=row["route"],
                form=row["form"],
            )

            gt_roles, err = get_ground_truth_roles(
                data, role_names, unii_to_roles,
                api_unii=row["api"],
                dose_mg=row["dose"],
                unit=row["unit"],
                route=row["route"],
                form=row["form"],
                dose_tol=args.dose_tol,
            )

            if err:
                print(format_output(results, args.threshold))
                print(f"  Ground truth: {err}\n")
            else:
                print(format_output(results, args.threshold, gt_roles=gt_roles))

                predicted = {r for r, p in results if p >= args.threshold}
                tp = len(predicted & gt_roles)
                fp = len(predicted - gt_roles)
                fn = len(gt_roles - predicted)
                precision = tp / (tp + fp + 1e-8)
                recall    = tp / (tp + fn + 1e-8)
                f1        = 2 * precision * recall / (precision + recall + 1e-8)
                batch_results.append({
                    "precision": precision, "recall": recall, "f1": f1
                })

        except ValueError as e:
            print(f"  Error: {e}")

        print()

    # Batch summary
    if len(batch_results) > 1:
        avg_p  = np.mean([r["precision"] for r in batch_results])
        avg_r  = np.mean([r["recall"]    for r in batch_results])
        avg_f1 = np.mean([r["f1"]        for r in batch_results])
        print(f"{'='*68}")
        print(f"  BATCH SUMMARY ({len(batch_results)} formulations with GT)")
        print(f"  Avg Precision: {avg_p:.3f}  "
              f"Avg Recall: {avg_r:.3f}  "
              f"Avg F1: {avg_f1:.3f}")
        print(f"{'='*68}")