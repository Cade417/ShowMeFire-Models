from __future__ import annotations

import torch
from torch import nn


QUANTILES = (0.1, 0.5, 0.9)


class StationSequenceModel(nn.Module):
    """GRU that predicts quantile corrections to the physics trajectory."""
    def __init__(self, input_size, hidden_size=16, num_layers=1, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_layers == 1 and dropout:
            raise ValueError("dropout is only supported for multilayer GRUs")
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
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
