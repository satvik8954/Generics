"""
model.py — MLP Role Predictor for ExciPick

Simple feedforward network:
    Input:  23-dim (api_features + dose + route + form)
    Output: 13-dim sigmoid (binary role probabilities)
"""

import torch
import torch.nn as nn


class RolePredictor(nn.Module):
    """
    Multi-label MLP that predicts which functional excipient roles
    are needed for a given API + formulation context.

    Args:
        input_dim:  feature dimension (default 23)
        num_roles:  number of role classes (default 13)
        hidden_dim: hidden layer size (default 128)
        dropout:    dropout rate (default 0.3)
    """

    def __init__(
        self,
        input_dim: int = 23,
        num_roles: int = 13,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),

            # Output
            nn.Linear(hidden_dim // 2, num_roles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) feature tensor
        Returns:
            probs: (B, num_roles) sigmoid probabilities
        """
        return self.net(x)
