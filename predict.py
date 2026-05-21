"""
predict.py — Inference script for ExciPick HetGNN.

Given an API (by UNII), dose, unit, route, and form,
predicts the most likely excipients for the formulation.

Usage:
    python predict.py --api 6CW7F3G59X --dose 500 --unit 1 --route ORAL --form TABLET
    python predict.py --api 6CW7F3G59X --dose 500 --unit 1 --route ORAL --form TABLET --show-gt
"""

import torch
import numpy as np
import pickle
import argparse
import csv
from config import CONFIG
from model.FULL_MODEL import ExciPickHGNN
from incompatibility_engine import IncompatibilityEngine

ORAL_ONLY_PATH = "Data/oral_only.csv"


def load_metadata():
    print("Loading metadata...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def load_default_from_oral_only():
    try:
        with open(ORAL_ONLY_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
    except FileNotFoundError:
        return None

    if not row:
        return None

    try:
        dose_val = float(row.get("dose_mg", ""))
    except ValueError:
        dose_val = None

    return {
        "api": row.get("api_unii") or None,
        "dose": dose_val,
        "unit": row.get("denominator_unit") or None,
        "route": row.get("route") or None,
        "form": row.get("primary_dosage_form") or None,
    }


def load_defaults_from_oral_only(limit):
    rows = []
    try:
        with open(ORAL_ONLY_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(rows) >= limit:
                    break
                try:
                    dose_val = float(row.get("dose_mg", ""))
                except ValueError:
                    dose_val = None
                rows.append(
                    {
                        "api": row.get("api_unii") or None,
                        "dose": dose_val,
                        "unit": row.get("denominator_unit") or None,
                        "route": row.get("route") or None,
                        "form": row.get("primary_dosage_form") or None,
                    }
                )
    except FileNotFoundError:
        return []

    return rows


def predict_excipients(
    model, graph, data, api_node_mapping, api_unii, dose_mg, unit, route, form, threshold=0.5
):
    device = next(model.parameters()).device
    model.eval()

    # 1. Get Vocabularies & Stats
    excipient_vocab = data["excipient_vocab"]
    route_vocab = data["route_vocab"]
    form_vocab = data["form_vocab"]
    per_unit_vocab = data["per_unit_vocab"]
    dose_stats = data["dose_stats"]

    # Reverse vocab to map IDs back to string names
    id_to_excipient = {v: k for k, v in excipient_vocab.items()}

    # 2. Look up API node index in graph
    if api_unii not in api_node_mapping:
        valid_examples = list(api_node_mapping.keys())[:3]
        raise ValueError(
            f"API UNII '{api_unii}' not found in the graph.\n"
            f"Some valid examples: {valid_examples}"
        )

    api_idx = torch.tensor([api_node_mapping[api_unii]], dtype=torch.long).to(device)

    # 3. Process Dose
    log_dose = np.log10(max(dose_mg, 1e-9))
    dose_normalized = (log_dose - dose_stats["mean"]) / dose_stats["std"]
    dose_tensor = torch.tensor([dose_normalized], dtype=torch.float32).to(device)

    # 4. Process Categorical IDs
    if unit not in per_unit_vocab:
        raise ValueError(f"Unit '{unit}' not found. Valid units: {list(per_unit_vocab.keys())}")
    if route not in route_vocab:
        raise ValueError(f"Route '{route}' not found. Valid routes: {list(route_vocab.keys())}")
    if form not in form_vocab:
        raise ValueError(f"Form '{form}' not found. Valid forms: {list(form_vocab.keys())[:10]}...")

    unit_tensor = torch.tensor([per_unit_vocab[unit]], dtype=torch.long).to(device)
    route_tensor = torch.tensor([route_vocab[route]], dtype=torch.long).to(device)
    form_tensor = torch.tensor([form_vocab[form]], dtype=torch.long).to(device)

    # 5. Model Forward Pass (with graph)
    with torch.no_grad():
        logits = model(graph, api_idx, dose_tensor, unit_tensor, route_tensor, form_tensor)
        probs = torch.sigmoid(logits)[0]

    # 6. Extract predictions above threshold
    predictions = []
    for exc_id in range(len(id_to_excipient)):
        prob = probs[exc_id].item()
        if prob >= threshold:
            predictions.append((id_to_excipient[exc_id], prob))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


def find_ground_truth(data, api_unii, dose_mg, unit, route, form, dose_tol=0.0, max_examples=5):
    df = data["df"]

    required_cols = {
        "api_unii",
        "dose_mg",
        "denominator_unit",
        "route",
        "primary_dosage_form",
        "excipient_ids",
    }
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        return None, f"Missing columns in processed data: {sorted(missing_cols)}"

    candidates = df[
        (df["api_unii"] == api_unii)
        & (df["route"] == route)
        & (df["primary_dosage_form"] == form)
        & (df["denominator_unit"] == unit)
    ]

    if candidates.empty:
        return None, "No matching formulation found for api/route/form/unit"

    if dose_tol <= 0:
        exact = candidates[candidates["dose_mg"] == dose_mg]
        if exact.empty:
            examples = (
                candidates["dose_mg"]
                .drop_duplicates()
                .sort_values()
                .head(max_examples)
                .tolist()
            )
            return None, (
                "No exact dose match (set --dose-tol to allow a tolerance). "
                f"Example doses: {examples}"
            )
        candidates = exact
    else:
        candidates = candidates[(candidates["dose_mg"] - dose_mg).abs() <= dose_tol]
        if candidates.empty:
            examples = (
                candidates["dose_mg"]
                .drop_duplicates()
                .sort_values()
                .head(max_examples)
                .tolist()
            )
            return None, (
                "No dose match within tolerance. "
                f"Example doses: {examples}"
            )

    # Use the closest dose if multiple rows
    candidates = candidates.copy()
    candidates["dose_diff"] = (candidates["dose_mg"] - dose_mg).abs()
    row = candidates.sort_values("dose_diff").iloc[0]

    return row["excipient_ids"], None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick HetGNN Inference")
    parser.add_argument("--api", type=str, default=None, help="API UNII code")
    parser.add_argument("--dose", type=float, default=None, help="Dose in mg")
    parser.add_argument("--unit", type=str, default=None, help="Denominator unit")
    parser.add_argument("--route", type=str, default=None, help="Route of administration")
    parser.add_argument("--form", type=str, default=None, help="Dosage form")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Model weights path")
    parser.add_argument("--show-gt", action="store_true", help="Print ground-truth excipients")
    parser.add_argument("--dose-tol", type=float, default=0.0, help="Dose tolerance for GT lookup")
    parser.add_argument(
        "--batch-from-oral",
        type=int,
        default=0,
        help="Run predictions for the first N rows in Data/oral_only.csv",
    )

    args = parser.parse_args()

    # Load metadata
    try:
        data = load_metadata()
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run preprocess.py first.")
        exit(1)

    input_rows = []
    if args.batch_from_oral > 0:
        input_rows = load_defaults_from_oral_only(args.batch_from_oral)
        if not input_rows:
            print("Error: Data/oral_only.csv not found or empty.")
            exit(1)
    else:
        defaults = load_default_from_oral_only()
        if defaults:
            if args.api is None:
                args.api = defaults["api"]
            if args.dose is None:
                args.dose = defaults["dose"]
            if args.unit is None:
                args.unit = defaults["unit"]
            if args.route is None:
                args.route = defaults["route"]
            if args.form is None:
                args.form = defaults["form"]

        missing_inputs = [
            name
            for name, value in {
                "api": args.api,
                "dose": args.dose,
                "unit": args.unit,
                "route": args.route,
                "form": args.form,
            }.items()
            if value in (None, "")
        ]
        if missing_inputs:
            print(f"Error: missing inputs: {missing_inputs}")
            print("Provide them via CLI or ensure Data/oral_only.csv has values.")
            exit(1)

        input_rows = [
            {
                "api": args.api,
                "dose": args.dose,
                "unit": args.unit,
                "route": args.route,
                "form": args.form,
            }
        ]

    vocab_size = len(data["excipient_vocab"])

    # Load graph
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        graph = torch.load("hetero_graph.pt", map_location=device, weights_only=False)
        graph = graph.to(device)
    except FileNotFoundError:
        print("Error: hetero_graph.pt not found. Run build_graph.py first.")
        exit(1)

    # Load API node mapping
    with open("api_node_mapping.pkl", "rb") as f:
        api_node_mapping = pickle.load(f)

    # Load model
    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Successfully loaded model weights from {args.model}")
    except FileNotFoundError:
        print(f"Warning: '{args.model}' not found. Using random weights for demo!")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        exit(1)

    # Load incompatibility engine
    engine = IncompatibilityEngine(
        handbook_flags_path="Data/incompatibilities_flags.json",
        features_db_path="Data/excipientsFeaturesDB.csv",
    )

    for idx, row in enumerate(input_rows, start=1):
        print("\n" + "=" * 50)
        if len(input_rows) > 1:
            print(f"INPUTS ({idx}/{len(input_rows)}):")
        else:
            print("INPUTS:")
        print(f"  API UNII : {row['api']}")
        print(f"  Dose     : {row['dose']} mg")
        print(f"  Unit     : {row['unit']}")
        print(f"  Route    : {row['route']}")
        print(f"  Form     : {row['form']}")
        print("=" * 50 + "\n")

        try:
            predictions = predict_excipients(
                model=model,
                graph=graph,
                data=data,
                api_node_mapping=api_node_mapping,
                api_unii=row["api"],
                dose_mg=row["dose"],
                unit=row["unit"],
                route=row["route"],
                form=row["form"],
                threshold=args.threshold,
            )
            # Get API SMILES for incompatibility checking
            api_smiles = None
            api_rows = data["df"][data["df"]["api_unii"] == row["api"]]
            if not api_rows.empty:
                api_smiles = api_rows.iloc[0].get("smiles", None)

            # Rerank with incompatibility engine
            reranked = engine.rerank(api_smiles, predictions)
            pred_names = [e["name"] for e in reranked]

            print(f"PREDICTED EXCIPIENTS (Threshold >= {args.threshold}):")
            if not reranked:
                print("  (No excipients predicted above the threshold)")
            else:
                print(engine.format_output(reranked))

            if args.show_gt:
                excipient_ids, err = find_ground_truth(
                    data=data,
                    api_unii=row["api"],
                    dose_mg=row["dose"],
                    unit=row["unit"],
                    route=row["route"],
                    form=row["form"],
                    dose_tol=args.dose_tol,
                )
                if err:
                    print(f"\nGROUND TRUTH: {err}")
                else:
                    id_to_excipient = {v: k for k, v in data["excipient_vocab"].items()}
                    gt_names = [id_to_excipient[eid] for eid in excipient_ids]
                    gt_set = set(gt_names)
                    pred_set = set(pred_names)
                    matched = sorted(gt_set & pred_set)

                    print("\nGROUND TRUTH EXCIPIENTS:")
                    for name in sorted(gt_set):
                        print(f"  - {name}")

                    print("\nMATCHED EXCIPIENTS:")
                    if matched:
                        for name in matched:
                            print(f"  - {name}")
                    else:
                        print("  (No matches)")

                    print(f"\nMatched {len(matched)}/{len(gt_set)} ground-truth excipients")

        except ValueError as e:
            print(f"Input Error: {e}")
