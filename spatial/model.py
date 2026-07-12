from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__(); self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, 3, padding=1)
    def forward(self, x, state):
        h, c = state; i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g); h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class SpatialQuantileModel(nn.Module):
    """ConvLSTM with versioned continuous and embedded categorical geography."""
    def __init__(self, dynamic_channels, static_continuous_channels, category_sizes, hidden_channels=32, embedding_dims=(4, 8, 8)):
        super().__init__()
        self.dynamic_channels = dynamic_channels; self.static_continuous_channels = static_continuous_channels
        self.category_sizes = tuple(category_sizes); self.embedding_dims = tuple(embedding_dims[:len(category_sizes)])
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0) for size, dim in zip(self.category_sizes, self.embedding_dims)])
        stem_inputs = dynamic_channels + static_continuous_channels + sum(self.embedding_dims)
        self.stem = nn.Sequential(nn.Conv2d(stem_inputs, hidden_channels, 5, stride=2, padding=2), nn.SiLU(),
                                  nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.SiLU())
        self.cell = ConvLSTMCell(hidden_channels, hidden_channels)
        self.head = nn.Sequential(nn.ConvTranspose2d(hidden_channels, 16, 4, stride=2, padding=1), nn.SiLU(),
                                  nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1))

    def _static(self, continuous, categorical):
        embedded = [embedding(categorical[:, index]).permute(0, 3, 1, 2)
                    for index, embedding in enumerate(self.embeddings)]
        return torch.cat([continuous, *embedded], dim=1) if embedded else continuous

    def forward(self, dynamic_sequence, static_continuous, static_categorical, physics):
        batch, steps = dynamic_sequence.shape[:2]; static = self._static(static_continuous, static_categorical)
        first = self.stem(torch.cat([dynamic_sequence[:, 0], static], dim=1))
        h = first.new_zeros((batch, self.cell.hidden_channels, *first.shape[-2:])); c = torch.zeros_like(h); output = []
        for step in range(steps):
            encoded = self.stem(torch.cat([dynamic_sequence[:, step], static], dim=1))
            h, c = self.cell(encoded, (h, c)); raw = self.head(h)
            median = raw[:, 0:1] + physics[:, step]
            output.append(torch.cat([median - torch.nn.functional.softplus(raw[:, 1:2]), median,
                                     median + torch.nn.functional.softplus(raw[:, 2:3])], dim=1))
        return torch.stack(output, dim=1)


def masked_spatial_pinball(prediction, target, mask, physics, smoothness_weight=1e-4, physics_weight=1e-4):
    quantiles = prediction.new_tensor((.1, .5, .9)).view(1, 1, 3, 1, 1); error = target - prediction
    observed = torch.maximum(quantiles * error, (quantiles - 1) * error) * mask
    observed_loss = observed.sum() / mask.sum().clamp_min(1) / 3; median = prediction[:, :, 1:2]
    smoothness = (median[..., 1:, :] - median[..., :-1, :]).abs().mean() + (median[..., :, 1:] - median[..., :, :-1]).abs().mean()
    return observed_loss + smoothness_weight * smoothness + physics_weight * (median - physics).abs().mean()
