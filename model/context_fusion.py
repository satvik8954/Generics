import torch
import torch.nn as nn
from config import CONFIG


class ContextFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.route_emb = nn.Embedding(
            CONFIG["route_vocab"],
            CONFIG["route_emb"]
        )

        self.form_emb = nn.Embedding(
            CONFIG["form_vocab"],
            CONFIG["form_emb"]
        )

        input_dim = (
            CONFIG["api_out"] +
            CONFIG["strength_out"] +
            CONFIG["route_emb"] +
            CONFIG["form_emb"]
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim, CONFIG["context_out"]),
            nn.ReLU()
        )

    def forward(self, api, strength, route, form):
        r = self.route_emb(route)
        f = self.form_emb(form)

        x = torch.cat([api, strength, r, f], dim=1)
        return self.net(x)