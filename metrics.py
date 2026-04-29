"""
metrics.py — Evaluation metrics for ExciPick.

Metrics:
  - Precision@K: fraction of top-K predictions that are correct
  - Recall@K: fraction of true excipients captured in top-K
  - F1@K: harmonic mean of Precision@K and Recall@K
  - Jaccard@K: intersection / union of predicted and true sets
"""

import torch
import numpy as np


def precision_at_k(predictions, targets, k=10):
    """
    Args:
        predictions: (B, V) raw logits or scores
        targets: (B, V) multi-hot binary targets
        k: number of top predictions to consider
    Returns:
        mean precision@k across batch
    """
    _, topk_idx = predictions.topk(k, dim=1)
    topk_hits = targets.gather(1, topk_idx)
    return topk_hits.sum(dim=1).float().div(k).mean().item()


def recall_at_k(predictions, targets, k=10):
    """Mean recall@k across batch."""
    _, topk_idx = predictions.topk(k, dim=1)
    topk_hits = targets.gather(1, topk_idx)
    true_counts = targets.sum(dim=1).clamp(min=1)
    return (topk_hits.sum(dim=1).float() / true_counts).mean().item()


def f1_at_k(predictions, targets, k=10):
    """Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(predictions, targets, k)
    r = recall_at_k(predictions, targets, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def jaccard_at_k(predictions, targets, k=10):
    """
    Mean Jaccard similarity between top-K predicted set and true set.
    Jaccard = |intersection| / |union|
    """
    _, topk_idx = predictions.topk(k, dim=1)

    # Build predicted set as multi-hot
    pred_set = torch.zeros_like(targets)
    pred_set.scatter_(1, topk_idx, 1.0)

    intersection = (pred_set * targets).sum(dim=1)
    union = ((pred_set + targets) > 0).float().sum(dim=1).clamp(min=1)

    return (intersection / union).mean().item()


def evaluate(model, dataloader, device, k_values=None):
    """
    Run full evaluation on a DataLoader.

    Returns a dict of metrics for each k value.
    """
    if k_values is None:
        k_values = [5, 10, 15]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            api = batch["api"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            output = model(api, dose, per_unit, route, form)

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


def print_metrics(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Group by k
    k_values = sorted(set(int(key.split("@")[1]) for key in results))
    metrics = ["Precision", "Recall", "F1", "Jaccard"]

    # Header
    header = f"{'Metric':<12}" + "".join(f"{'@'+str(k):>10}" for k in k_values)
    print(header)
    print("-" * len(header))

    for metric in metrics:
        row = f"{metric:<12}"
        for k in k_values:
            val = results.get(f"{metric}@{k}", 0)
            row += f"{val:>10.4f}"
        print(row)

    print("=" * 50)
