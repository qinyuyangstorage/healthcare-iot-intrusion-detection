from __future__ import annotations

import torch
from torch import nn


class LightweightIDS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)
