"""
train.py — Training loop for ExciPick Role Predictor

Usage:
    python role_predictor/train.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import RolePredictor

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH  = "role_predictor/role_dataset.pkl"
MODEL_SAVE    = "role_predictor/best_role_model.pt"

INPUT_DIM     = 23
HIDDEN_DIM    = 128
DROPOUT       = 0.3
LR            = 1e-3
BATCH_SIZE    = 256
EPOCHS        = 50
PATIENCE      = 50
SEED          = 42

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def compute_f1(preds, targets, threshold=0.5):
    """Compute macro F1 across all roles."""
    pred_bin = (preds >= threshold).float()
    
    tp = (pred_bin * targets).sum(dim=0)
    fp = (pred_bin * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_bin) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.mean().item(), f1


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs("role_predictor", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load dataset
    print("\n[1] Loading dataset...")
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    X_train = torch.tensor(dataset["X_train"], dtype=torch.float32)
    y_train = torch.tensor(dataset["y_train"], dtype=torch.float32)
    X_val   = torch.tensor(dataset["X_val"],   dtype=torch.float32)
    y_val   = torch.tensor(dataset["y_val"],   dtype=torch.float32)

    role_names = dataset["role_names"]
    num_roles  = dataset["num_roles"]

    print(f"    Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"    Roles: {num_roles}")

    # 2. DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 3. Model
    model = RolePredictor(
        input_dim=INPUT_DIM,
        num_roles=num_roles,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    print(f"\n[2] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Loss + Optimizer
    # Use pos_weight to handle class imbalance
    # roles like coating_agent appear in ~30% of formulations
    # roles like humectant appear in ~3% — needs upweighting
    role_counts = dataset["role_counts"]
    total = len(dataset["X_train"])
    pos_weights = []
    for role in role_names:
        count = role_counts.get(role, 1)
        weight = (total - count) / max(count, 1)
        pos_weights.append(min(weight, 10.0))  # cap at 10x

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
   

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Training loop
    print("\n[3] Training...")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                val_losses.append(loss.item())
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        avg_val_loss = np.mean(val_losses)
        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_f1, _   = compute_f1(all_preds, all_targets)

        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  "
            f"Val F1: {val_f1:.4f}  "
            f"LR: {lr:.2e}"
        )

        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_f1   = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"    * saved (val_loss={best_val_loss:.4f}, val_f1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs.")
                break

    print(f"\n[OK] Training complete.")
    print(f"     Best val loss: {best_val_loss:.4f}")
    print(f"     Best val F1:   {best_val_f1:.4f}")
    print(f"     Model saved:   {MODEL_SAVE}")


if __name__ == "__main__":
    main()
