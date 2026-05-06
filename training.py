"""
training.py — Full training pipeline for ExciPick HetGNN.

Pipeline:
  1. Load preprocessed data + heterogeneous graph + split
  2. Create DataLoaders
  3. Train with GNN-aware forward pass + validation loop
  4. Save best model
"""

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

from dataset import ExciDataset
from model.FULL_MODEL import ExciPickHGNN
from config import CONFIG


def main():
    # ─────────────────────────────────────────
    # SEED
    # ─────────────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ─────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────
    print("Loading preprocessed data...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    excipient_vocab = data["excipient_vocab"]
    vocab_size = len(excipient_vocab)

    print("Loading split data...")
    with open("split_data.pkl", "rb") as f:
        split_data = pickle.load(f)

    train_df = split_data["train_df"]
    val_df = split_data["val_df"]
    test_df = split_data["test_df"]

    print(f"  Dataset splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Excipient vocab: {vocab_size}")

    # ─────────────────────────────────────────
    # LOAD GRAPH + API MAPPING
    # ─────────────────────────────────────────
    print("\nLoading heterogeneous graph...")
    graph = torch.load("hetero_graph.pt", weights_only=False)
    print(f"  API nodes: {graph['api'].num_nodes}")
    print(f"  Excipient nodes: {graph['excipient'].num_nodes}")

    with open("api_node_mapping.pkl", "rb") as f:
        api_node_mapping = pickle.load(f)

    # ─────────────────────────────────────────
    # DATASETS + DATALOADERS
    # ─────────────────────────────────────────
    train_dataset = ExciDataset(train_df, vocab_size, api_node_mapping)
    val_dataset = ExciDataset(val_df, vocab_size, api_node_mapping)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    # ─────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Move graph to device
    graph = graph.to(device)

    model = ExciPickHGNN(
        graph_metadata=graph.metadata(),
        vocab_size=vocab_size,
    ).to(device)

    # Dummy forward pass to initialize lazy SAGEConv parameters
    with torch.no_grad():
        dummy_idx = torch.zeros(1, dtype=torch.long, device=device)
        dummy_dose = torch.zeros(1, device=device)
        dummy_cat = torch.zeros(1, dtype=torch.long, device=device)
        model(graph, dummy_idx, dummy_dose, dummy_cat, dummy_cat, dummy_cat)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
    )

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )

    # Class imbalance is handled dynamically by Focal Loss
    # We no longer need the extreme pos_weight approach
    print("  Using Focal Loss (alpha=0.25, gamma=2.0) instead of standard BCE.")

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # ─────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\nTraining for {CONFIG['epochs']} epochs...\n")

    for epoch in range(CONFIG["epochs"]):

        # --- Train ---
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False):
            api_idx = batch["api_idx"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(graph, api_idx, dose, per_unit, route, form)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", leave=False):
                api_idx = batch["api_idx"].to(device)
                dose = batch["dose"].to(device)
                per_unit = batch["per_unit"].to(device)
                route = batch["route"].to(device)
                form = batch["form"].to(device)
                target = batch["target"].to(device)

                output = model(graph, api_idx, dose, per_unit, route, form)
                loss = loss_fn(output, target)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Logging ---
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']}  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"LR: {current_lr:.6f}", end="")

        # --- Save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("  * saved", end="")

        print()

    print(f"\n[OK] Training complete. Best val loss: {best_val_loss:.4f}")
    print("   Model saved to: best_model.pt")

    # ─────────────────────────────────────────
    # SAVE TEST SET FOR EVALUATION
    # ─────────────────────────────────────────
    with open("test_data.pkl", "wb") as f:
        pickle.dump({
            "test_df": test_df,
            "excipient_vocab": excipient_vocab,
            "vocab_size": vocab_size,
        }, f)
    print("   Test data saved to: test_data.pkl")


if __name__ == "__main__":
    main()