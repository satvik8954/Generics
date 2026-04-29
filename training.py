import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ExciDataset
from models.full_model import ExciPickModel
from config import CONFIG

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Data/f3.csv")

# 🔴 IMPORTANT: make sure excipients column is list
# if string → convert first

# =========================
# BUILD VOCABS
# =========================

routes = df["route"].unique()
route2id = {r: i for i, r in enumerate(routes)}

forms = df["primary_dosage_form"].unique()
form2id = {f: i for i, f in enumerate(forms)}

from collections import Counter

all_excipients = []
for ex in df["excipients"]:
    all_excipients.extend(ex)

counts = Counter(all_excipients)
vocab = [e for e, c in counts.items() if c >= 3]

excipient2id = {e: i for i, e in enumerate(vocab)}

# =========================
# DATASET + DATALOADER
# =========================

dataset = ExciDataset(df, route2id, form2id, excipient2id)

train_loader = DataLoader(
    dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True
)

# =========================
# MODEL
# =========================

device = CONFIG["device"]

model = ExciPickModel(vocab_size=len(excipient2id)).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG["lr"]
)

loss_fn = torch.nn.BCEWithLogitsLoss()

# =========================
# TRAIN LOOP
# =========================

for epoch in range(CONFIG["epochs"]):

    model.train()
    total_loss = 0

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

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")