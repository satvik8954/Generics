"""
test.py — Evaluate the trained ExciPick model on the test set.

Usage:
  python test.py                     # evaluate saved model
  python test.py --smoke             # quick smoke test with dummy data
"""

import argparse
import pickle
import torch
from torch.utils.data import DataLoader

from model.FULL_MODEL import ExciPickModel
from dataset import ExciDataset
from metrics import evaluate, print_metrics
from config import CONFIG


def smoke_test():
    """Quick forward-pass check with dummy data."""
    print("Running smoke test...")
    model = ExciPickModel(vocab_size=1000)

    api = torch.randn(4, CONFIG["api_in"])
    dose = torch.randn(4)
    per_unit = torch.randint(0, 3, (4,))
    route = torch.randint(0, 10, (4,))
    form = torch.randint(0, 10, (4,))

    output = model(api, dose, per_unit, route, form)
    print(f"[OK] Smoke test passed! Output shape: {output.shape}")


def run_evaluation():
    """Load best model + test data and compute metrics."""
    print("Loading test data...")
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    test_df = test_data["test_df"]
    excipient_vocab = test_data["excipient_vocab"]
    vocab_size = test_data["vocab_size"]

    print(f"  Test set: {len(test_df)} rows")
    print(f"  Vocab size: {vocab_size}")

    # Dataset + Loader
    test_dataset = ExciDataset(test_df, vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Load model
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    model = ExciPickModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    print(f"  Loaded best_model.pt on {device}")

    # Evaluate
    k_values = [5, CONFIG["top_k"], 15]
    results = evaluate(model, test_loader, device, k_values=k_values)
    print_metrics(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        run_evaluation()