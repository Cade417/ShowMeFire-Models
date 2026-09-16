"""Guarded seven-quantile residual GRU for station model V4."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class GuardedQuantileGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, residual_cap=4.0):
        super().__init__(); self.residual_cap = float(residual_cap)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 8))

    def forward(self, features, base, lead_weights=None, return_gate=False):
        raw = self.head(self.gru(features)[0]); gate = torch.sigmoid(raw[..., 7])
        correction = self.residual_cap * torch.tanh(raw[..., 0]) * gate
        if lead_weights is not None: correction = correction * lead_weights
        median = base + correction; distance = F.softplus(raw[..., 1:7]) + 1e-4
        lower25 = median - distance[..., 2]
        lower10 = lower25 - distance[..., 1]
        lower05 = lower10 - distance[..., 0]
        upper75 = median + distance[..., 3]
        upper90 = upper75 + distance[..., 4]
        upper95 = upper90 + distance[..., 5]
        output = torch.stack((lower05, lower10, lower25, median, upper75, upper90, upper95), dim=-1)
        return (output, gate) if return_gate else output


def v4_loss(prediction, gate, target, mask, gate_regularization=0.01, sample_weight=None):
    observed = mask > 0; actual = target[observed]; predicted = prediction[observed]
    if not actual.numel(): return prediction.sum() * 0
    weight = torch.ones_like(actual)
    if sample_weight is not None:
        weight = weight * sample_weight[observed]
    weight = torch.where(actual <= 6, 1.5 * weight, weight)
    threshold = torch.minimum(torch.minimum((actual - 7).abs(), (actual - 9).abs()), (actual - 15).abs())
    weight = torch.where(threshold <= 1, 1.25 * weight, weight)
    error = predicted[:, 3] - actual
    p50 = (weight * F.huber_loss(predicted[:, 3], actual, reduction="none", delta=2.0)).mean()
    mse = (weight * error.square()).mean()
    pinball = 0.0
    for index, quantile in enumerate(QUANTILES):
        difference = actual - predicted[:, index]
        pinball = pinball + torch.maximum(quantile * difference, (quantile - 1) * difference).mean()
    bias = error.mean().abs()
    return p50 + 0.15 * mse + 0.30 * pinball / len(QUANTILES) + 0.05 * bias + gate_regularization * gate[observed].mean()
