"""Ordered-quantile GRU that corrects a causal XGBoost base trajectory."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HybridStationModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers == 1 and dropout:
            raise ValueError("dropout must be zero for a one-layer GRU")
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 3))

    def forward(self, features, base_fm):
        hidden, _ = self.gru(features)
        raw = self.head(hidden)
        p50 = base_fm + raw[..., 0]
        lower = F.softplus(raw[..., 1]) + 1e-4
        upper = F.softplus(raw[..., 2]) + 1e-4
        return torch.stack((p50 - lower, p50, p50 + upper), dim=-1)


def hybrid_loss(prediction, target, mask, p50_loss="mae", quantile_weight=0.3, bias_weight=0.05):
    denominator = mask.sum().clamp_min(1)
    median_error = prediction[..., 1] - target
    if p50_loss == "huber":
        median = F.smooth_l1_loss(prediction[..., 1], target, reduction="none")
    elif p50_loss == "mae":
        median = median_error.abs()
    else:
        raise ValueError(f"Unsupported P50 loss: {p50_loss}")
    median = (median * mask).sum() / denominator
    outer_prediction = prediction[..., (0, 2)]
    q = prediction.new_tensor((0.1, 0.9)).view(1, 1, 2)
    error = target.unsqueeze(-1) - outer_prediction
    pinball = torch.maximum(q * error, (q - 1) * error)
    pinball = (pinball * mask.unsqueeze(-1)).sum() / denominator / 2
    bias = ((median_error * mask).sum() / denominator).abs()
    return median + quantile_weight * pinball + bias_weight * bias
