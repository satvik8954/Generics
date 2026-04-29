import torch
import torch.nn as nn
from config import CONFIG


class StrengthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.per_unit_emb = nn.Embedding(
            CONFIG["per_unit_vocab"],
            CONFIG["per_unit_emb"]
        )

        self.net = nn.Sequential(
            nn.Linear(1 + CONFIG["per_unit_emb"], CONFIG["strength_out"]),
            nn.ReLU()
        )

    def forward(self, dose, per_unit):
        emb = self.per_unit_emb(per_unit)
        x = torch.cat([dose.unsqueeze(1), emb], dim=1)
        return self.net(x)