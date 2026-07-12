from __future__ import annotations

import torch
from torch import nn


QUANTILES = (0.1, 0.5, 0.9)


class StationSequenceModel(nn.Module):
    """GRU that predicts quantile corrections to the physics trajectory."""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 3))

    def forward(self, features, physics_fm):
        hidden, _ = self.gru(features)
        correction = self.head(hidden)
        return physics_fm.unsqueeze(-1) + correction


def pinball_loss(prediction, target, mask=None):
    q = prediction.new_tensor(QUANTILES).view(1, 1, 3)
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(q * error, (q - 1) * error)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
        return loss.sum() / mask.sum().clamp_min(1) / 3
    return loss.mean()
