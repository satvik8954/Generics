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
    graph = torch.load("hetero_graph.pt")
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

    # Class imbalance: compute pos_weight from training data
    avg_positives = train_df["excipient_ids"].apply(len).mean()
    pw = (vocab_size - avg_positives) / avg_positives
    pos_weight = torch.tensor([pw], device=device)
    print(f"  pos_weight: {pw:.1f} (avg {avg_positives:.1f} positives per sample out of {vocab_size})")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

        for batch in train_loader:
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
            for batch in val_loader:
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