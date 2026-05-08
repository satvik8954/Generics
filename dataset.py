"""
dataset.py — PyTorch Dataset for ExciPick HetGNN model.

Each sample returns:
  - api_idx:  scalar long — index of the API node in the heterogeneous graph
  - dose:     scalar float — normalized log dose
  - per_unit: scalar long — per-unit ID
  - route:    scalar long — route ID
  - form:     scalar long — dosage form ID
  - target:   (vocab_size,) float tensor — multi-hot excipient labels
"""

import torch
from torch.utils.data import Dataset


class ExciDataset(Dataset):
    def __init__(self, df, excipient_vocab_size, api_node_mapping):
        """
        Args:
            df: preprocessed DataFrame with columns:
                api_unii, dose_normalized, per_unit_id, route_id, form_id, excipient_ids
            excipient_vocab_size: total number of excipients in vocabulary
            api_node_mapping: dict mapping api_unii -> graph node index
        """
        self.df = df.reset_index(drop=True)
        self.vocab_size = excipient_vocab_size
        self.api_node_mapping = api_node_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # API node index in the heterogeneous graph
        api_idx = torch.tensor(
            self.api_node_mapping[row["api_unii"]], dtype=torch.long
        )

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
            "api_idx": api_idx,
            "dose": dose,
            "per_unit": per_unit,
            "route": route,
            "form": form,
            "target": target,
        }
