import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseProjection(nn.Module):
    """Projection head for context and excipient embeddings."""
    def __init__(self, in_dim=256, out_dim=128, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SiameseNetwork(nn.Module):
    """
    Siamese-style model that maps context vectors and excipient embeddings
    into a common embedding space for contrastive learning.
    """
    def __init__(self, context_dim=256, exc_dim=256, proj_dim=128, hidden=256, dropout=0.1):
        super().__init__()

        # Separate input projections (modal-specific) followed by shared projection
        self.context_input = nn.Linear(context_dim, hidden)
        self.exc_input = nn.Linear(exc_dim, hidden)

        # Shared projection head
        self.proj = SiameseProjection(in_dim=hidden, out_dim=proj_dim, hidden=hidden, dropout=dropout)

    def forward(self, context, exc):
        """
        Args:
            context: (B, context_dim)
            exc:     (B, exc_dim)

        Returns:
            z_ctx: (B, proj_dim)
            z_exc: (B, proj_dim)
        """
        h_ctx = F.relu(self.context_input(context))
        h_exc = F.relu(self.exc_input(exc))

        z_ctx = self.proj(h_ctx)
        z_exc = self.proj(h_exc)

        # Normalize for contrastive distance computations
        z_ctx = F.normalize(z_ctx, p=2, dim=1)
        z_exc = F.normalize(z_exc, p=2, dim=1)

        return z_ctx, z_exc


class ContrastiveLoss(nn.Module):
    """
    Hadsell et al. contrastive loss:
      L = y * D^2 + (1 - y) * max(0, margin - D)^2
    where y=1 for positive pairs
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, labels):
        # Euclidean distance
        diff = z1 - z2
        dist = torch.norm(diff, p=2, dim=1)

        pos_loss = labels.float() * (dist ** 2)
        neg_loss = (1 - labels.float()) * (F.relu(self.margin - dist) ** 2)

        loss = pos_loss + neg_loss
        return loss.mean()
