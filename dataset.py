"""
dataset.py — PyTorch Dataset for ExciPick model.

Each sample returns:
  - api:      (20,) float tensor — normalized molecular descriptors
  - dose:     scalar float — normalized log dose
  - per_unit: scalar long — per-unit ID
  - route:    scalar long — route ID
  - form:     scalar long — dosage form ID
  - target:   (vocab_size,) float tensor — multi-hot excipient labels
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class ExciDataset(Dataset):
    def __init__(self, df, excipient_vocab_size):
        """
        Args:
            df: preprocessed DataFrame with columns:
                api_features, dose_normalized, per_unit_id, route_id, form_id, excipient_ids
            excipient_vocab_size: total number of excipients in vocabulary
        """
        self.df = df.reset_index(drop=True)
        self.vocab_size = excipient_vocab_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # API molecular descriptors (20-dim normalized vector)
        api = torch.tensor(row["api_features"], dtype=torch.float32)

        # Normalized dose
        dose = torch.tensor(row["dose_normalized"], dtype=torch.float32)

        # Categorical IDs
        per_unit = torch.tensor(row["per_unit_id"], dtype=torch.long)
        route = torch.tensor(row["route_id"], dtype=torch.long)
        form = torch.tensor(row["form_id"], dtype=torch.long)

        # Multi-hot target over excipient vocab
        target = torch.zeros(self.vocab_size, dtype=torch.float32)
        for eid in row["excipient_ids"]:
            target[eid] = 1.0

        return {
            "api": api,
            "dose": dose,
            "per_unit": per_unit,
            "route": route,
            "form": form,
            "target": target,
        }
