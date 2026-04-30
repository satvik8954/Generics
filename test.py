"""
test.py — Evaluate the trained ExciPick HetGNN model on the test set.

Usage:
  python test.py                     # evaluate saved model
  python test.py --smoke             # quick smoke test with dummy data
"""

import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from model.FULL_MODEL import ExciPickHGNN
from dataset import ExciDataset
from metrics import evaluate, print_metrics
from config import CONFIG


def smoke_test():
    """Quick forward-pass check with dummy graph and data."""
    print("Running smoke test...")

    vocab_size = 100

    # Build a minimal dummy graph
    graph = HeteroData()
    graph["api"].x = torch.randn(10, CONFIG["api_in"])
    graph["api"].num_nodes = 10
    graph["excipient"].num_nodes = vocab_size

    graph["api", "uses", "excipient"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    graph["excipient", "used_by", "api"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    graph["excipient", "cooccurs", "excipient"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )
    graph["api", "similar", "api"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )

    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    )

    api_idx = torch.randint(0, 10, (4,))
    dose = torch.randn(4)
    per_unit = torch.randint(0, 3, (4,))
    route = torch.randint(0, 10, (4,))
    form = torch.randint(0, 10, (4,))

    output = model(graph, api_idx, dose, per_unit, route, form)
    print(f"[OK] Smoke test passed! Output shape: {output.shape}")
    assert output.shape == (4, vocab_size), f"Expected (4, {vocab_size}), got {output.shape}"


def run_evaluation():
    """Load best model + graph + test data and compute metrics."""
    print("Loading test data...")
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    test_df = test_data["test_df"]
    excipient_vocab = test_data["excipient_vocab"]
    vocab_size = test_data["vocab_size"]

    print(f"  Test set: {len(test_df)} rows")
    print(f"  Vocab size: {vocab_size}")

    # Load API node mapping
    with open("api_node_mapping.pkl", "rb") as f:
        api_node_mapping = pickle.load(f)

    # Dataset + Loader
    test_dataset = ExciDataset(test_df, vocab_size, api_node_mapping)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )

    # Load graph
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    graph = torch.load("hetero_graph.pt", map_location=device)
    graph = graph.to(device)

    # Load model
    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    ).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    print(f"  Loaded best_model.pt on {device}")

    # Evaluate using graph-aware forward pass
    k_values = [5, CONFIG["top_k"], 15]
    results = evaluate_with_graph(model, graph, test_loader, device, k_values)
    print_metrics(results)


def evaluate_with_graph(model, graph, dataloader, device, k_values=None):
    """Run full evaluation with graph-aware forward pass."""
    from metrics import precision_at_k, recall_at_k, f1_at_k, jaccard_at_k

    if k_values is None:
        k_values = [5, 10, 15]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            api_idx = batch["api_idx"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            output = model(graph, api_idx, dose, per_unit, route, form)

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    results = {}
    for k in k_values:
        results[f"Precision@{k}"] = precision_at_k(preds, targets, k)
        results[f"Recall@{k}"] = recall_at_k(preds, targets, k)
        results[f"F1@{k}"] = f1_at_k(preds, targets, k)
        results[f"Jaccard@{k}"] = jaccard_at_k(preds, targets, k)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExciPick HetGNN evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        run_evaluation()