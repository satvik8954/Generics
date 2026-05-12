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
from config import CONFIG
from model.FULL_MODEL import ExciPickHGNN


def load_metadata():
    print("Loading metadata...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data


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


def find_ground_truth(data, api_unii, dose_mg, unit, route, form, dose_tol=0.0):
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
            return None, "No exact dose match (set --dose-tol to allow a tolerance)"
        candidates = exact
    else:
        candidates = candidates[(candidates["dose_mg"] - dose_mg).abs() <= dose_tol]
        if candidates.empty:
            return None, "No dose match within tolerance"

    # Use the closest dose if multiple rows
    candidates = candidates.copy()
    candidates["dose_diff"] = (candidates["dose_mg"] - dose_mg).abs()
    row = candidates.sort_values("dose_diff").iloc[0]

    return row["excipient_ids"], None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick HetGNN Inference")
    parser.add_argument("--api", type=str, default="6CW7F3G59X", help="API UNII code")
    parser.add_argument("--dose", type=float, default=100.0, help="Dose in mg")
    parser.add_argument("--unit", type=str, default="1", help="Denominator unit")
    parser.add_argument("--route", type=str, default="ORAL", help="Route of administration")
    parser.add_argument("--form", type=str, default="TABLET", help="Dosage form")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Model weights path")
    parser.add_argument("--show-gt", action="store_true", help="Print ground-truth excipients")
    parser.add_argument("--dose-tol", type=float, default=0.0, help="Dose tolerance for GT lookup")

    args = parser.parse_args()

    # Load metadata
    try:
        data = load_metadata()
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Run preprocess.py first.")
        exit(1)

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

    print("\n" + "=" * 50)
    print("INPUTS:")
    print(f"  API UNII : {args.api}")
    print(f"  Dose     : {args.dose} mg")
    print(f"  Unit     : {args.unit}")
    print(f"  Route    : {args.route}")
    print(f"  Form     : {args.form}")
    print("=" * 50 + "\n")

    try:
        predictions = predict_excipients(
            model=model,
            graph=graph,
            data=data,
            api_node_mapping=api_node_mapping,
            api_unii=args.api,
            dose_mg=args.dose,
            unit=args.unit,
            route=args.route,
            form=args.form,
            threshold=args.threshold,
        )
        pred_names = [name for name, _ in predictions]

        print(f"PREDICTED EXCIPIENTS (Threshold >= {args.threshold}):")
        if not predictions:
            print("  (No excipients predicted above the threshold)")
        else:
            for name, prob in predictions:
                print(f"  - {name:<30} ({prob:.1%} confidence)")

        if args.show_gt:
            excipient_ids, err = find_ground_truth(
                data=data,
                api_unii=args.api,
                dose_mg=args.dose,
                unit=args.unit,
                route=args.route,
                form=args.form,
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
