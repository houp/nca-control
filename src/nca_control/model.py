from __future__ import annotations

import torch
from torch import nn


class ControllableNCAModel(nn.Module):
    """A minimal NCA-style model for controlled one-step state updates."""

    def __init__(
        self,
        input_channels: int = 6,
        state_channels: int = 1,
        hidden_channels: int = 32,
        cell_value_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_channels = state_channels
        self.cell_value_max = cell_value_max

        self.perception = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, channels, height, width]")
        if inputs.shape[1] < self.state_channels:
            raise ValueError("inputs do not contain enough state channels")

        current_state = inputs[:, : self.state_channels]
        features = self.perception(inputs)
        delta = self.update(features)
        return torch.sigmoid(current_state + delta) * self.cell_value_max
