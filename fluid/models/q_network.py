from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class MinAtarQNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(num_linear_units, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        return f.relu(self.fc_hidden(x))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        q_values = self.q_head(features)
        if return_features:
            return q_values, features
        return q_values
