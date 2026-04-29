"""
training.py — Full training pipeline for ExciPick.

Pipeline:
  1. Load preprocessed data (from preprocess.py)
  2. Split into train/val/test (from split.py)
  3. Create DataLoaders
  4. Train with validation loop
  5. Save best model
"""

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ExciDataset
from model.FULL_MODEL import ExciPickModel
from split import split_by_api_cluster
from config import CONFIG


def main():
    # ─────────────────────────────────────────
    # SEED
    # ─────────────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ─────────────────────────────────────────
    # LOAD PREPROCESSED DATA
    # ─────────────────────────────────────────
    print("Loading preprocessed data...")
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    df = data["df"]
    excipient_vocab = data["excipient_vocab"]
    vocab_size = len(excipient_vocab)

    print(f"  Dataset: {len(df)} rows")
    print(f"  Excipient vocab: {vocab_size}")

    # ─────────────────────────────────────────
    # TRAIN / VAL / TEST SPLIT
    # ─────────────────────────────────────────
    print("\nSplitting data...")
    train_df, val_df, test_df = split_by_api_cluster(df, seed=CONFIG["seed"])
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ─────────────────────────────────────────
    # DATASETS + DATALOADERS
    # ─────────────────────────────────────────
    train_dataset = ExciDataset(train_df, vocab_size)
    val_dataset = ExciDataset(val_df, vocab_size)
    test_dataset = ExciDataset(test_df, vocab_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    # ─────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model = ExciPickModel(vocab_size=vocab_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

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
            api = batch["api"].to(device)
            dose = batch["dose"].to(device)
            per_unit = batch["per_unit"].to(device)
            route = batch["route"].to(device)
            form = batch["form"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(api, dose, per_unit, route, form)
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
                api = batch["api"].to(device)
                dose = batch["dose"].to(device)
                per_unit = batch["per_unit"].to(device)
                route = batch["route"].to(device)
                form = batch["form"].to(device)
                target = batch["target"].to(device)

                output = model(api, dose, per_unit, route, form)
                loss = loss_fn(output, target)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # --- Logging ---
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']}  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}", end="")

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