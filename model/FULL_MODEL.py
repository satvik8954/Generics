import torch.nn as nn

from model.api_encoder import APIEncoder
from model.strength_encoder import StrengthEncoder
from model.context_fusion import ContextFusion
from model.excipient_scorer import Scorer


class ExciPickModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.api_encoder = APIEncoder()
        self.strength_encoder = StrengthEncoder()
        self.context_fusion = ContextFusion()
        self.scorer = Scorer(vocab_size)

    def forward(self, api_feat, dose, per_unit, route, form):
        api_out = self.api_encoder(api_feat)
        strength_out = self.strength_encoder(dose, per_unit)

        context = self.context_fusion(
            api_out,
            strength_out,
            route,
            form
        )

        scores = self.scorer(context)

        return scores