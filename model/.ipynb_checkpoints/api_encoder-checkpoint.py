import torch.nn as nn
from config import CONFIG


class APIEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(CONFIG["api_in"], CONFIG["api_hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["api_hidden"], CONFIG["api_out"])
        )

    def forward(self, x):
        return self.net(x)