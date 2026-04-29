import torch
import torch.nn as nn
from config import CONFIG


class Scorer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.excipient_emb = nn.Embedding(
            vocab_size,
            CONFIG["excipient_emb"]
        )

        input_dim = CONFIG["context_out"] + CONFIG["excipient_emb"]

        self.net = nn.Sequential(
            nn.Linear(input_dim, CONFIG["scorer_hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["scorer_hidden"], 1)
        )

    def forward(self, context):
        B = context.shape[0]
        V = self.excipient_emb.num_embeddings

        exc_emb = self.excipient_emb.weight  # (V, emb_dim)

        context = context.unsqueeze(1).repeat(1, V, 1)
        exc_emb = exc_emb.unsqueeze(0).repeat(B, 1, 1)

        x = torch.cat([context, exc_emb], dim=2)
        scores = self.net(x).squeeze(-1)

        return scores