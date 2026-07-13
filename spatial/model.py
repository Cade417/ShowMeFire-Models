from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__(); self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, 3, padding=1)
    def forward(self, x, state):
        h, c = state; i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class TeacherStudentSpatialModel(nn.Module):
    """Shared causal encoder with realized-RTMA teacher and forecast-HRRR student."""
    target_leads = tuple(range(4, 16))

    def __init__(self, static_continuous_channels, category_sizes, hidden_channels=32, embedding_dims=(4, 8, 8)):
        super().__init__(); self.static_continuous_channels = static_continuous_channels
        self.category_sizes = tuple(category_sizes); self.embedding_dims = tuple(embedding_dims[:len(category_sizes)])
        self.hidden_channels = hidden_channels
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0)
                                         for size, dim in zip(self.category_sizes, self.embedding_dims)])
        context_channels = 5 + static_continuous_channels + sum(self.embedding_dims)
        self.context_stem = self._stem(context_channels, hidden_channels)
        self.antecedent_stem = self._stem(4, hidden_channels)
        self.teacher_stem = self._stem(4, hidden_channels)
        self.student_stem = self._stem(4, hidden_channels)
        self.antecedent_cell = ConvLSTMCell(hidden_channels, hidden_channels)
        self.teacher_cell = ConvLSTMCell(hidden_channels, hidden_channels)
        self.student_cell = ConvLSTMCell(hidden_channels, hidden_channels)
        self.head = nn.Sequential(nn.ConvTranspose2d(hidden_channels, 16, 4, stride=2, padding=1), nn.SiLU(),
                                  nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1))

    @staticmethod
    def _stem(inputs, hidden):
        return nn.Sequential(nn.Conv2d(inputs, hidden, 5, stride=2, padding=2), nn.SiLU(),
                             nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU())

    def _static(self, continuous, categorical):
        embedded = [embedding(categorical[:, index]).permute(0, 3, 1, 2)
                    for index, embedding in enumerate(self.embeddings)]
        return torch.cat([continuous, *embedded], dim=1) if embedded else continuous

    def encode_causal(self, antecedent, current, static_continuous, static_categorical):
        context = self.context_stem(torch.cat([current, self._static(static_continuous, static_categorical)], dim=1))
        h, c = context, torch.zeros_like(context)
        for step in range(antecedent.shape[1]):
            h, c = self.antecedent_cell(self.antecedent_stem(antecedent[:, step]), (h, c))
        return h, c

    def _quantiles(self, hidden, physics):
        raw = self.head(hidden); median = raw[:, 0:1] + physics
        return torch.cat((median - F.softplus(raw[:, 1:2]), median, median + F.softplus(raw[:, 2:3])), dim=1)

    def student_forward(self, antecedent, hrrr, current, static_continuous, static_categorical, physics):
        h, c = self.encode_causal(antecedent, current, static_continuous, static_categorical)
        output, hidden = [], []
        for step in range(hrrr.shape[1]):
            h, c = self.student_cell(self.student_stem(hrrr[:, step]), (h, c)); hidden.append(h)
            output.append(self._quantiles(h, physics[:, step]))
        return torch.stack(output, 1), torch.stack(hidden, 1)

    def forward(self, antecedent, realized, hrrr, current, static_continuous, static_categorical, physics):
        causal_h, causal_c = self.encode_causal(antecedent, current, static_continuous, static_categorical)
        teacher_h, teacher_c = causal_h, causal_c; teacher_output, teacher_hidden = [], []
        for hour in range(realized.shape[1]):
            teacher_h, teacher_c = self.teacher_cell(self.teacher_stem(realized[:, hour]), (teacher_h, teacher_c))
            if hour + 1 in self.target_leads:
                index = self.target_leads.index(hour + 1); teacher_hidden.append(teacher_h)
                teacher_output.append(self._quantiles(teacher_h, physics[:, index]))
        student_h, student_c = causal_h, causal_c; student_output, student_hidden = [], []
        for step in range(hrrr.shape[1]):
            student_h, student_c = self.student_cell(self.student_stem(hrrr[:, step]), (student_h, student_c))
            student_hidden.append(student_h); student_output.append(self._quantiles(student_h, physics[:, step]))
        return (torch.stack(student_output, 1), torch.stack(teacher_output, 1),
                torch.stack(student_hidden, 1), torch.stack(teacher_hidden, 1))


class StudentExport(nn.Module):
    """Student-only inference surface; the teacher is unreachable from ONNX."""
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, antecedent_rtma, hrrr_forecast, current_fm_state, static_continuous,
                static_categorical, physics_trajectory):
        return self.model.student_forward(antecedent_rtma, hrrr_forecast, current_fm_state,
                                          static_continuous, static_categorical, physics_trajectory)[0]


def masked_pinball(prediction, target, mask):
    quantiles = prediction.new_tensor((.1, .5, .9)).view(1, 1, 3, 1, 1)
    error = target - prediction
    return (torch.maximum(quantiles * error, (quantiles - 1) * error) * mask).sum() / mask.sum().clamp_min(1) / 3


def teacher_student_loss(student, teacher, student_hidden, teacher_hidden, target, mask, physics,
                         distilled=True, smoothness_weight=1e-4, physics_weight=1e-4):
    loss = masked_pinball(student, target, mask)
    if distilled:
        loss = loss + masked_pinball(teacher, target, mask)
        loss = loss + .25 * (((student[:, :, 1:2] - teacher[:, :, 1:2].detach()).abs() * mask).sum() / mask.sum().clamp_min(1))
        loss = loss + .10 * F.mse_loss(student_hidden, teacher_hidden.detach())
    median = student[:, :, 1:2]
    smoothness = (median[..., 1:, :] - median[..., :-1, :]).abs().mean() + (median[..., :, 1:] - median[..., :, :-1]).abs().mean()
    return loss + smoothness_weight * smoothness + physics_weight * (median - physics).abs().mean()
