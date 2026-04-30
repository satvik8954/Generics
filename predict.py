import torch
import numpy as np
import pandas as pd
import pickle
import argparse
from config import CONFIG
from model.FULL_MODEL import ExciPickModel

def load_metadata(pkl_path="processed_data.pkl"):
    print(f"Loading metadata from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

def predict_excipients(model, data, api_unii, dose_mg, unit, route, form, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()

    # 1. Get Vocabularies & Stats
    df = data["df"]
    excipient_vocab = data["excipient_vocab"]
    route_vocab = data["route_vocab"]
    form_vocab = data["form_vocab"]
    per_unit_vocab = data["per_unit_vocab"]
    dose_stats = data["dose_stats"]
    
    # Reverse vocab to map IDs back to string names
    id_to_excipient = {v: k for k, v in excipient_vocab.items()}

    # 2. Look up API Features
    # Find the API in the processed dataframe to get its raw features
    api_rows = df[df["api_unii"] == api_unii]
    if len(api_rows) == 0:
        raise ValueError(f"API UNII '{api_unii}' not found in the dataset. Please provide a known UNII.")
    
    # Get the normalized API features directly from the dataframe
    api_features_normalized = api_rows.iloc[0]["api_features"]
    api_tensor = torch.tensor([api_features_normalized], dtype=torch.float32).to(device)

    # 3. Process Dose
    log_dose = np.log10(max(dose_mg, 1e-9))
    dose_normalized = (log_dose - dose_stats["mean"]) / dose_stats["std"]
    dose_tensor = torch.tensor([dose_normalized], dtype=torch.float32).to(device)

    # 4. Process Categorical IDs
    if unit not in per_unit_vocab:
        raise ValueError(f"Unit '{unit}' not found. Valid units: {list(per_unit_vocab.keys())[:5]}...")
    if route not in route_vocab:
        raise ValueError(f"Route '{route}' not found. Valid routes: {list(route_vocab.keys())[:5]}...")
    if form not in form_vocab:
        raise ValueError(f"Form '{form}' not found. Valid forms: {list(form_vocab.keys())[:5]}...")

    unit_tensor = torch.tensor([per_unit_vocab[unit]], dtype=torch.long).to(device)
    route_tensor = torch.tensor([route_vocab[route]], dtype=torch.long).to(device)
    form_tensor = torch.tensor([form_vocab[form]], dtype=torch.long).to(device)

    # 5. Model Forward Pass
    with torch.no_grad():
        logits = model(api_tensor, dose_tensor, unit_tensor, route_tensor, form_tensor)
        probs = torch.sigmoid(logits)[0]  # Get batch index 0

    # 6. Extract predictions above threshold
    predictions = []
    for exc_id in range(len(id_to_excipient)):
        prob = probs[exc_id].item()
        if prob >= threshold:
            predictions.append((id_to_excipient[exc_id], prob))

    # Sort by probability descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick Inference Script")
    parser.add_argument("--api", type=str, default="O819R91D7A", help="API UNII code (must exist in dataset)")
    parser.add_argument("--dose", type=float, default=100.0, help="Dose in mg")
    parser.add_argument("--unit", type=str, default="1", help="Denominator unit (e.g., '1', 'mL', 'TABLET')")
    parser.add_argument("--route", type=str, default="ORAL", help="Route of administration")
    parser.add_argument("--form", type=str, default="TABLET", help="Dosage form")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold for prediction")
    parser.add_argument("--model", type=str, default="best_model.pt", help="Path to trained model weights")
    
    args = parser.parse_args()

    # Load Metadata
    try:
        data = load_metadata()
    except FileNotFoundError:
        print("Error: processed_data.pkl not found. Please run preprocess.py first.")
        exit(1)

    vocab_size = len(data["excipient_vocab"])
    
    # Setup Device & Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ExciPickModel(vocab_size=vocab_size).to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Successfully loaded model weights from {args.model}")
    except FileNotFoundError:
        print(f"Warning: Model weights '{args.model}' not found. Using randomly initialized weights for demonstration!")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        exit(1)

    print("\n" + "="*50)
    print("INPUTS:")
    print(f"  API UNII : {args.api}")
    print(f"  Dose     : {args.dose} mg")
    print(f"  Unit     : {args.unit}")
    print(f"  Route    : {args.route}")
    print(f"  Form     : {args.form}")
    print("="*50 + "\n")

    try:
        predictions = predict_excipients(
            model=model,
            data=data,
            api_unii=args.api,
            dose_mg=args.dose,
            unit=args.unit,
            route=args.route,
            form=args.form,
            threshold=args.threshold
        )
        
        print(f"PREDICTED EXCIPIENTS (Threshold >= {args.threshold}):")
        if not predictions:
            print("  (No excipients predicted above the threshold)")
        else:
            for name, prob in predictions:
                print(f"  - {name:<30} ({prob:.1%} confidence)")
                
    except ValueError as e:
        print(f"Input Error: {e}")
        print("\nTip: Check processed_data.pkl or run without arguments to see a valid example.")
